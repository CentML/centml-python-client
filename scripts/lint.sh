#!/bin/bash -e

# run pylint
python -m pylint --rcfile ./scripts/pylintrc ./centml ./tests
