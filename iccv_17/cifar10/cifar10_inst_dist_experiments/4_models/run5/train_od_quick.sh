#!/usr/bin/env sh

# TOOLS=./build/tools
TOOLS=/home/mike/ml_lab/mod/MCL_Caffe/build/tools

$TOOLS/caffe train \
  --solver=/home/mike/ml_lab/mod/MCL_Caffe/examples/cifar10/cifar10_inst_dist_experiments/4_models/run5/mcl_cifar10_quick_solver.prototxt \
  --gpu 0 2>&1 | tee -a /home/mike/ml_lab/mod/MCL_Caffe/examples/cifar10/cifar10_inst_dist_experiments/4_models/run5/output_4000.log
# reduce learning rate by factor of 10 after 8 epochs
$TOOLS/caffe train \
  --solver=/home/mike/ml_lab/mod/MCL_Caffe/examples/cifar10/cifar10_inst_dist_experiments/4_models/run5/mcl_cifar10_quick_solver_lr1.prototxt \
  --gpu 0 \
  --snapshot=/home/mike/ml_lab/mod/MCL_Caffe/examples/cifar10/cifar10_inst_dist_experiments/4_models/run5/mcl_cifar10_quick_iter_4000.solverstate 2>&1 | tee -a /home/mike/ml_lab/mod/MCL_Caffe/examples/cifar10/cifar10_inst_dist_experiments/4_models/run5/output_5000.log
