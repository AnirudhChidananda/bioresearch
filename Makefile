.PHONY: help install sync test lint server playground eval-gold eval-random eval-heuristic eval-all docker clean

PYTHON ?= uv run python
UV ?= uv

help:
	@echo "Bioresearch — available make targets"
	@echo "  make sync          — install all dev + playground deps"
	@echo "  make test          — run the full test suite"
	@echo "  make server        — launch the OpenEnv FastAPI server on port 8000"
	@echo "  make playground    — launch the Gradio playground on port 7860"
	@echo "  make eval-gold     — run the gold-policy ceiling evaluation"
	@echo "  make eval-random   — run the random-policy floor evaluation"
	@echo "  make eval-heuristic— run the token-overlap heuristic baseline"
	@echo "  make eval-all      — run all three baselines and write eval_*.json"
	@echo "  make docker        — build the HF-Spaces-ready Docker image"
	@echo "  make clean         — remove __pycache__, .pytest_cache, dist/"

sync:
	$(UV) sync --extra dev --extra playground

install: sync

test:
	$(UV) run pytest -q

lint:
	@echo "No linter configured yet — pytest is the contract."

server:
	$(UV) run uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

playground:
	$(UV) run python playground.py

eval-gold:
	$(PYTHON) evaluate.py --policy gold --episodes 10 --output eval_gold.json

eval-random:
	$(PYTHON) evaluate.py --policy random --episodes 10 --output eval_random.json

eval-heuristic:
	$(PYTHON) evaluate.py --policy heuristic --episodes 10 --output eval_heuristic.json

eval-all: eval-random eval-heuristic eval-gold
	@echo "All baselines written to eval_*.json"

docker:
	docker build -t bioresearch:latest .

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__ .pytest_cache dist build *.egg-info
	find . -name "*.pyc" -delete
