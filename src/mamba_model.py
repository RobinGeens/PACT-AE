import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from src.config import Mamba1Config
from src.named_ops import DummyEinsum, NamedAdd, NamedEinsum, NamedEinsumSingleOp, NamedMul
from src.ssm import ssm1_op
from src.util import Stage


# taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, use_mup: bool = False):
        super().__init__()  # type: ignore

        self.use_mup = use_mup
        self.eps = 1e-5

        # https://arxiv.org/abs/2404.05728, RMSNorm gains prevents muTransfer (section 4.2.3)
        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor):
        rstd = 1 / torch.sqrt(x.pow(2).mean(-1, keepdim=True))  # + self.eps) Skip eps step for simplicity
        output = x * rstd

        if not self.use_mup:
            return output * self.weight
        else:
            return output


class Mamba1Block(nn.Module):
    def __init__(self, config: Mamba1Config, stage: Stage):
        super().__init__()  # type: ignore

        self.config = config
        self.stage = stage

        # Constants
        self.batch = config.batch_size
        self.L = config.prefill_size
        self.d_model = config.d_model
        self.ED = config.d_inner
        self.H = config.n_head
        self.d_head = config.d_head
        self.N = config.d_state * config.n_groups
        self.d_conv = config.d_conv  # kernel size e.g. 4
        self.conv_channels = self.ED + 2 * self.N
        self.d_in_proj = 2 * self.ED + 2 * self.N + self.H

        # Operators
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.do_in_proj_bias)
        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.do_conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        self.out_proj = nn.Linear(self.ED, self.d_model, bias=self.config.do_out_proj_bias)
        self.residual = NamedAdd()
        self.norm = RMSNorm(self.d_model, self.config.use_mup)
        self.act = nn.SiLU()

        # The variable name will be the module name in ONNX, e.g. `mul_dBx/Einsum`
        self.mul_y_z = NamedMul()
        self.dt_proj_einsum = NamedEinsum()
        self.mul_delta_A = NamedEinsum()
        self.mul_delta_B = NamedEinsum()
        self.mul_dBx = NamedEinsum()
        self.mul_conv = NamedEinsum()
        self.reduce_conv = NamedEinsumSingleOp()
        self.update_states = DummyEinsum()

        self.mul_delta_B_step = NamedMul()
        self.mul_dBx_step = NamedMul()
        self.add_delta_t = NamedAdd()
        self.mul_Ah = NamedEinsum()
        self.mul_h_C = NamedEinsum()
        self.add_dBx = NamedAdd()
        self.mul_D_x = NamedEinsum()
        self.add_y_Dx = NamedAdd()

        # Parameters
        dt = torch.exp(torch.rand(config.d_inner) * (math.log(0.1) - math.log(1e-3)) + math.log(0.1))
        dt = torch.clamp(dt, min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        A_log = nn.Parameter(torch.log(A))
        self.A = nn.Parameter(-torch.exp(A_log))
        self.D = nn.Parameter(torch.ones(config.d_inner))

    def get_empty_cache(self):
        """
        state_cache: (B, ED, N)
        conv_cache: (B, ED, d_conv-1)
        """
        state_cache = torch.zeros(
            self.batch,
            self.ED,
            self.N,
        )
        conv_cache = torch.zeros(
            self.batch,
            self.ED,
            self.d_conv - 1,
        )
        return (state_cache, conv_cache)

    def forward(self, x: Tensor):
        """
        x: (B, L, D)
        out: (B, L, D)
        """
        with torch.no_grad():
            u = self.norm(x)

            match self.stage:
                case Stage.PREFILL:
                    out, _ = self._forward(u)
                case Stage.DECODE:
                    # out, _ = self.step(u, self.get_empty_cache())
                    # Use standard forward instead of step as there seems to be dimension mismatch
                    out, _ = self._forward(u)

            out = self.residual(out, x)
            return out

    def _forward(self, x: Tensor):
        """
        x: (B, L, D)
        out: (B, L, D)
        """
        _, L, _ = x.shape

        xz = self.in_proj(x)  # (B, L, 2*ED)
        x, z = torch.split(xz, [self.ED, self.ED], dim=-1)

        # x branch
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)  # (B, L + padding,  ED)
        x = x[:, :L]
        x = self.act(x)

        deltaBC = self.x_proj(x)  # (B, L, dt_rank+2*N)
        dt, B, C = torch.split(  # (B, L, dt_rank), (B, L, N), (B, L, N)
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )

        dt = self.dt_proj_einsum(  # (ED, dt_rank) @ (B, L, dt_rank) -> (B, L, ED)
            "blh,dh->bld", dt, self.dt_proj.weight
        )
        dt = F.softplus(dt + self.dt_proj.bias)  # (B, L, ED)
        y, state = self.selective_scan(x, dt, self.A, B, C)

        # z branch
        z = self.act(z)

        out = self.mul_y_z(y, z)  # (B, L, ED)
        out = self.out_proj(out)  # (B, L, D)
        return out, state

    def selective_scan(self, x: Tensor, dt: Tensor, A: Tensor, B: Tensor, C: Tensor):
        """
        x: (B, L, ED)
        dt: (B, L, ED)
        A: (ED, N)
        B: (B, L, N)
        C: (B, L, N)
        D: (ED,)
        out: (B, L, ED)
        """

        dA = self.mul_delta_A("bld,dn->bldn", dt, A)  # (B, L, ED) * (ED, N) -> (B, L, ED, N)
        dA = torch.exp(dA)

        dB = self.mul_delta_B("bld,bln->bldn", dt, B)  # (B, L, ED) * (B, L, N) -> (B, L, ED, N)
        dBx = self.mul_dBx("bldn,bld->bldn", dB, x)  # (B, L, ED, N) * (B, L, ED) -> (B, L, ED, N)

        h_init = self.get_empty_cache()[0]
        y, h = ssm1_op(dA, dBx, C, h_init)  # (B, L, ED, N)

        Dx = self.mul_D_x("bld,d->bld", x, self.D)  # (B, L, ED) * (ED) -> (B, L, ED)
        y = self.add_y_Dx(y, Dx)  # (B, L, ED)
        return y, h

    def update_state_iterative(self, dA: Tensor, dBx: Tensor, C: Tensor):
        """This is a naive and iterative implementation of pscan in O(n) instead of O(log(n))
        NOTE this is the worst possible implementation since the memory requirement for the intermediate state is
        O(n) i.s.o. O(1).
        The memory requirement of y is O(n) by default since y is needed for each timestep

        dA: (B, L, ED, N)
        dBx: (B, L, ED, N)
        C: (B, L, N)

        """
        # Compute all intermediate states and reduce in L dimension
        # NOTE this requires memory to save all intermediate states, whereas the ref implementation computes in-place
        states: Tensor = self.update_states("bldn,bldn->bldn", dA, dBx)
        states = states.cumsum(dim=1)  # (B, L, ED, N
        last_state = states[:, -1, :, :]

        # Compute y for all time steps
        y = self.mul_h_C("bldn,bln->bld", states, C)

        return y, last_state

    def update_state_iterative_inplace(self, dA: Tensor, dBx: Tensor, C: Tensor):
        """
        dA: (B, L, ED, N)
        dBx: (B, L, ED, N)
        C: (B, L, N)
        h: (B, ED, N)
        y: (B, L, ED)
        """
        state = self.get_empty_cache()[0]
        y = torch.zeros((self.batch, 0, self.ED))

        for i in range(self.L):
            state = self.mul_delta_A("bdn,bdn->bdn", state, dA[:, i, :, :])
            state = self.add_dBx(state, dBx[:, i, :, :])
            y_i = self.mul_h_C("bdn,bn->bd", state, C[:, i, :])
            y_i = rearrange(y_i, "b d -> b 1 d")
            y = torch.concat((y, y_i), dim=1)

        return y, state

    def step(self, x: Tensor, cache: tuple[Tensor, Tensor]):
        """
        x: (B, D)
        cache: (state_cache, conv_cache)
        state_cache: (B, ED, N)
        conv_cache: (B, ED, d_conv-1)
        out: (B, D)
        """

        state_cache, conv_cache = cache

        xz = self.in_proj(x)  # (B, 2*ED)
        x, z = torch.split(xz, [self.ED, self.ED], dim=-1)  # (B, ED), (B, ED)

        # x branch
        # Seems to me like the oldest sample is not kept in the cache already so we don't need to discard it
        # conv_cache = conv_cache[:, :, 1 : self.d_conv]  # Discard oldest sample

        x_unsqueezed = x.view(self.batch, self.ED, 1)
        conv_cache = torch.cat([conv_cache, x_unsqueezed], dim=-1)  # (B, ED, d_conv)
        conv = self.mul_conv("bdw,dlw->bdw", conv_cache, self.conv1d.weight)
        # Reduce in d_conv dim
        conv = self.reduce_conv("bdw->bd", conv)  # (B, ED)

        x = self.act(conv)
        deltaBC = self.x_proj(x)  # (B, dt_rank+2*N)

        dt, B, C = torch.split(  # (B, dt_rank), (B, N), (B, N)
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )

        y, state_cache = self.ssm_step(x, dt, B, C, state_cache)

        # z branch
        z = self.act(z)
        out = self.mul_y_z(y, z)
        out = self.out_proj(out)  # (B, D)

        return out, (state_cache, conv_cache)

    def ssm_step(self, x: Tensor, dt: Tensor, B: Tensor, C: Tensor, state_cache: Tensor):
        """
        x: (B, ED)
        dt: (B, dt_rank)
        B: (B, N)
        C: (B, N)
        A: (ED, N)
        D: (ED,)
        state_cache: (B, ED, N)
        out: (B, ED)
        """
        dt = self.dt_proj(dt)  # (B, ED)
        dt = F.softplus(dt)

        dA = self.mul_delta_A("bd,dn->bdn", dt, self.A)  # (B, ED) * (ED, N) -> (B, ED, N)
        dA = torch.exp(dA)

        dB = self.mul_delta_B("bd,bn->bdn", dt, B)  # (B, ED) * (B, N) -> (B, ED, N)
        dBx = self.mul_dBx("bdn,bd->bdn", dB, x)

        Ah = self.mul_Ah("bdn,bdn->bdn", state_cache, dA)
        h = Ah + dBx  # (B, ED, N)

        y = self.mul_h_C("bdn,bn->bd", h, C)
        Dx = self.mul_D_x("d,bd->bd", self.D, x)

        y = self.add_y_Dx(y, Dx)

        return y, h
