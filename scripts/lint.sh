#!/bin/bash -e

# run pylint
python -m pylint --rcfile ./scripts/pylintrc -j $(nproc) ./centml ./tests
