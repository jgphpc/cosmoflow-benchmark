#!/bin/bash -l
#SBATCH --constraint=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -t 0:20:00
#SBATCH -J train-cosmoflow-daint
#SBATCH -o logs/%x-%j.out

. scripts/setup_daint.sh

# TODO: output logs and horovod timeline to output directory
#export HOROVOD_TIMELINE=./timeline.json

# Configuration
#nTrain=262144
#nValid=65536
#sourceDir=/global/cscratch1/sd/sfarrell/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf
dataDir=/scratch/snx3000/lukasd/mlperf/data/cosmoflow/cosmoUniverse_2019_05_4parE_tf

# Data staging skipped

# Run the training
set -x
srun -l -u python train.py --distributed --rank-gpu --data-dir ${dataDir} $@
