#!/usr/bin/env bash

set -x
set -e
set -u
set -o pipefail

# usage:
# nohup bash build.sh >/dev/null 2>&1 &
# tail -f nohup-gpu.out
# tail -f nohup-cpu.out

docker build --build-arg BUILD_DATE="$(TZ=UTC-8 date "+%Y-%m-%dT%H:%M:%S+08:00")" -t duruyao/caffe:gpu --load gpu --progress plain >nohup-gpu.out 2>&1
docker push duruyao/caffe:gpu

docker build --build-arg BUILD_DATE="$(TZ=UTC-8 date "+%Y-%m-%dT%H:%M:%S+08:00")" -t duruyao/caffe:cpu --load cpu --progress plain >nohup-cpu.out 2>&1
docker push duruyao/caffe:cpu

