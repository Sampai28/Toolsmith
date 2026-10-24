# Toolsmith.
#
#   make up && make all
#
# Everything runs in containers. Nothing here installs on the host, contacts a
# hosted API, or touches version control.

SHELL   := /bin/sh
COMPOSE := docker compose -f docker/docker-compose.yml
CONFIG  ?= configs/quick.yaml
MODEL   ?= qwen2.5:7b-instruct

# One-shot container on the Compose network, so TOOLSMITH_OLLAMA_URL resolves.
RUN := $(COMPOSE) run --rm -T harness

.DEFAULT_GOAL := help

## help: list targets
help:
	@echo "Toolsmith:"
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /'
	@echo ""
	@echo "  CONFIG=$(CONFIG)  MODEL=$(MODEL)"

## up: start ollama, api and dashboard; pull the model if absent
up:
	$(COMPOSE) up -d --build ollama
	@echo "waiting for ollama..."
	@$(COMPOSE) exec -T ollama sh -c 'until ollama list >/dev/null 2>&1; do sleep 1; done'
	@$(COMPOSE) exec -T ollama sh -c 'ollama list | grep -q "$(MODEL)" || ollama pull $(MODEL)'
	$(COMPOSE) up -d --build api app
	@echo "api        http://localhost:8000/healthz"
	@echo "dashboard  http://localhost:8501"

## down: stop everything and remove volumes
down:
	$(COMPOSE) down -v

## pull: fetch the model into the ollama volume
pull:
	$(COMPOSE) exec -T ollama ollama pull $(MODEL)

## generate: build the synthetic corpus and run the dataset validation gates
generate:
	$(RUN) toolsmith.data.generate --config $(CONFIG)
	$(RUN) toolsmith.data.validate --strict

## validate: run the task-suite validation gates
validate:
	$(RUN) toolsmith.harness.cli validate --config $(CONFIG)

## determinism: assert an identical config reproduces identical results
determinism:
	$(RUN) toolsmith.harness.cli determinism --config $(CONFIG)

## baseline: STEP 0 - can the model do the task at all? Gates the ablation.
baseline:
	$(RUN) toolsmith.harness.cli baseline --config $(CONFIG)

## ablate: run the two-factor grid (refuses if the baseline gate failed)
ablate:
	$(RUN) toolsmith.harness.cli ablate --config $(CONFIG)

## adversarial: run the attack suite and catalogue bypasses
adversarial:
	$(RUN) toolsmith.adversarial.run --config $(CONFIG)

## report: build reports/evaluation.html from the committed results
report:
	$(RUN) toolsmith.report.build

## serve: start the FastAPI service on :8000
serve:
	$(COMPOSE) up -d --build api
	@echo "api http://localhost:8000/healthz"

## dashboard: start the Streamlit dashboard on :8501
dashboard:
	$(COMPOSE) up -d --build app
	@echo "dashboard http://localhost:8501"

## test: run the pytest suite in a container
test:
	$(COMPOSE) run --rm -T --entrypoint pytest harness -q

## all: generate, validate, baseline, ablate, adversarial, report
all: generate validate determinism baseline ablate adversarial report
	@echo ""
	@echo "done. report: reports/evaluation.html"

.PHONY: help up down pull generate validate determinism baseline ablate \
        adversarial report serve dashboard test all
