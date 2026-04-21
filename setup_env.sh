uv venv .venv
source .venv/bin/activate
uv sync
uv pip install -e .
uv pip install jupyter ipykernel
python -m ipykernel install --user --name data-science-notebooks --display-name "Python (data science notebooks)"