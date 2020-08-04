#!/bin/bash

set -ex

timestamp=$(date "+%Y-%m-%d_%H-%M-%S")
output_dir_postfix="test" #"${timestamp}_${HOSTNAME}"

for i in $(seq 0 3); do 
    n_ranks=$((2**i))
    mpiexec -np ${n_ranks} python3 train.py --data-benchmark --distributed \
      --data-dir ../data/cosmoflow-test/ \
      --output-dir "results/data_benchmark/${output_dir_postfix}/gpu-n${n_ranks}" \
      --n-train $((8*${n_ranks})) --n-valid $((8*${n_ranks})) --n-epochs 10 \
      --verbose configs/cosmo_local.yaml
done
