#!/bin/bash
# build the project

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

if ! $WITH_CMAKE ; then
  echo -e "\e[35m\e[1mmake --jobs $NUM_THREADS all test pycaffe warn\e[0m"
  make --jobs $NUM_THREADS all test pycaffe warn
else
  echo -e "\e[35m\e[1mcd build; make --jobs $NUM_THREADS all test.testbin\e[0m"
  cd build
  make --jobs $NUM_THREADS all test.testbin
fi
echo -e "\e[35m\e[1mmake lint\e[0m"
make lint
