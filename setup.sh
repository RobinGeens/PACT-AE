python3.11 -m venv env
source env/bin/activate
pip install -r requirements.txt
git submodule update --init --recursive
python -m build stream
pip install stream/dist/*.whl
