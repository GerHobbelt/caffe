#!/bin/bash


nvlink_configs="0,1 0,1,2,3 0,1,2,6,5,4"
pci_configs="0,5 0,5,2,7 0,5,2,4,3,6"
nets="bvlc_alexnet  bvlc_googlenet  bvlc_reference_caffenet  inception-v3 resnet-50  vgg-16"

caffe_run() {
  # net solver num cfg outfile
  echo "Running $1 $2 $3 $4" >> $5 2>&1
  (time ./build/tools/caffe train --solver=$2 --gpu $4) >> $5 2>&1
}

#solver="caffe-models/bvlc_alexnet/solver.prototxt"
#./mod_solver.sh 100 $solver
#caffe_run bvlc_alexnet $solver 100 0,1 test.log

num=100
inc=100
until [ "$num" -gt "10000" ]; do
  for net in $nets; do
    solver="caffe-models/$net/solver.prototxt"
    ./mod_solver.sh $num $solver
    #cat $solver
    for cfg in $nvlink_configs; do
      caffe_run $net $solver $num $cfg "nvlink.log"
    done
    for cfg in $pci_configs; do
      caffe_run $net $solver $num $cfg "pci.log"
    done
  done
  num=$((num + inc))
  if [ "$num" == "1000" ]; then
    inc=1000
  fi
done
