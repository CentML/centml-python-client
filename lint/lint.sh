#!/bin/bash

# work in the same directory of this script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd $SCRIPT_DIR

# source ../env/bin/activate

# run pylint
python -m pylint --rcfile ./pylintrc -j $(nproc) ../centml
python -m pylint --rcfile ./pylintrc -j $(nproc) ../tests
