#!/bin/bash -e

# run black formatter
python -m black --skip-string-normalization --skip-magic-trailing-comma --line-length 120 ./centml ./tests $*
