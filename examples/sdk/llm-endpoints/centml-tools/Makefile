CONFIG ?= test_config.json
NAME ?=

PYTHON := /opt/homebrew/bin/python3.11


.PHONY: help deploy delete inspect

help:
	@echo "Available commands:"
	@echo "  make deploy CONFIG=<config_file.json>"
	@echo "  make delete CONFIG=<config_file.json>"
	@echo "  make inspect NAME=<model_name> or CONFIG=<config_file.json>"

deploy:
	@echo "Using Python: $(PYTHON)"
	$(PYTHON) scripts/deploy_model.py $(CONFIG)

delete:
	$(PYTHON) scripts/delete_deployment.py $(CONFIG)

inspect:
	$(PYTHON) scripts/inspect_model.py $(CONFIG)
