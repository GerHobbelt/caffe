#!/bin/bash
# install extra Python dependencies
# (must come after setup-venv)

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

# Python3
pip install --pre protobuf==3.0.0b3
pip install pydot
