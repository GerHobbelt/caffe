#!/bin/bash
# install extra Python dependencies
# (must come after setup-venv)

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

# Python3
pip install \
  pydot
