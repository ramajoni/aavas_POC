# The repo runs without being pip-installed: PYTHONPATH points at src/ so every
# target works straight after a clone.
PYTHON ?= python3
# The OmniXAI stack pins dash/Flask older than the rest of the environment, so it
# lives in an overlay venv built by scripts/setup_xai_env.sh.
XAI_PYTHON ?= $(CURDIR)/.venv-xai/bin/python
export PYTHONPATH := $(CURDIR)/src

.PHONY: install data train demo test api lint docker-up docker-phase2 all clean xai xai-install xai-reports automl automl-status monitors verify-algorithms

install:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) data/synthetic/generate_dataset.py

train:
	$(PYTHON) scripts/train_all.py

demo:
	$(PYTHON) scripts/run_demo.py --fresh-store

verify-algorithms:
	$(PYTHON) scripts/verify_algorithms.py $(ARGS)

monitors:
	$(PYTHON) scripts/run_monitors.py $(ARGS)

test:
	$(PYTHON) -m pytest -q

api:
	$(PYTHON) -m uvicorn aavas.api.main:app --reload --port 8000

notebooks:
	$(PYTHON) -m jupyter lab notebooks/

# Agentic AutoML search, recorded in MLflow. Selects and explains a champion;
# it does not promote one - models/ still comes from scripts/train_all.py.
automl:
	$(PYTHON) scripts/run_automl.py $(ARGS)

automl-status:
	$(PYTHON) scripts/run_automl.py --status

xai-install:
	bash scripts/setup_xai_env.sh

# Global + instance explainability for every trained model, with the dashboards.
xai:
	$(XAI_PYTHON) scripts/run_omnixai.py $(ARGS)

# Same analysis, reports only - no server, nothing to Ctrl-C.
xai-reports:
	$(XAI_PYTHON) scripts/run_omnixai.py --no-ui $(ARGS)

docker-up:
	docker compose up --build

docker-phase2:
	docker compose --profile phase2 up --build

# Full cold start: data -> models -> tests -> walkthrough.
all: data train test demo

clean:
	rm -rf models/*.joblib models/*.json data/raw/*.csv data/processed/*.csv *.db
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
