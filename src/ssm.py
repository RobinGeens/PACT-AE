"""
This file defines two SSM operators for Mamba1 and Mamba2
Their difference lies in the dimensions of the input tensors
"""

import torch
from torch._custom_op import impl as custom_op


# Mamba 1 SSM OP
@custom_op.custom_op("mylibrary::ssm1_op")
def ssm1_op(
    dA: torch.Tensor, dBx: torch.Tensor, C: torch.Tensor, h: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...
@ssm1_op.impl("cpu")
def ssm1_op_impl(dA, dBx, C, h):
    with torch.no_grad():
        B, L, ED, N = dA.shape
        y = []

        for t in range(L):
            h = dA[:, t] * h + dBx[:, t]
            y_t = torch.einsum("bdn,bn->bd", h, C[:, t])
            y.append(y_t)

        y = torch.stack(y, dim=1)  # Shape: (B, L, ED)
        return y, h


# Mamba 2 SSM OP
@custom_op.custom_op("mylibrary::ssm2_op")
def ssm2_op(
    dA: torch.Tensor, dBx: torch.Tensor, C: torch.Tensor, h: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...
@ssm2_op.impl("cpu")
def ssm2_op_impl(dA, dBx, C, h):
    with torch.no_grad():
        B, L, H = dA.shape
        y = []

        for t in range(L):
            h = torch.einsum("bh,bhdn->bhdn", dA[:, t, :], h) + dBx[:, t, :]
            y_t = torch.einsum("bhdn,bn->bhd", h, C[:, t, :])
            y.append(y_t)

        y = torch.stack(y, dim=1)
        return y, h
