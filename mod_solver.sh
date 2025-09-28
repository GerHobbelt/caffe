#!/bin/bash

val=$1
filename=$2

configs="test_iter test_interval stepsize max_iter snapshot"

for config in $configs; do
  sed -i "s/^${config}: .*/${config}: ${val}/" "${filename}"
done
