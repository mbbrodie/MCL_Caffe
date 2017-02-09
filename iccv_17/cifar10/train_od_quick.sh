#!/usr/bin/env sh

# TOOLS=./build/tools
TOOLS=/home/mike/ml_lab/mod/MCL_Caffe/build/tools

$TOOLS/caffe train \
  --solver=/home/mike/ml_lab/mod/MCL_Caffe/examples/cifar10/mcl_cifar10_quick_solver.prototxt \
  --gpu 0
# reduce learning rate by factor of 10 after 8 epochs
$TOOLS/caffe train \
  --solver=/home/mike/ml_lab/mod/MCL_Caffe/examples/cifar10/mcl_cifar10_quick_solver_lr1.prototxt \
  --gpu 0 
