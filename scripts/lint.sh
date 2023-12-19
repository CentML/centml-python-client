#!/bin/bash -e

# run pylint
python -m pylint --rcfile ./pylintrc -j $(nproc) ./centml ./tests
