#!/bin/bash

set -ex

timestamp=$(date "+%Y-%m-%d_%H-%M-%S")
output_dir_postfix="${timestamp}_${HOSTNAME}"

# Scaling locally
for log_n_ranks in $(seq 0 2); do
  n_ranks=$((2**log_n_ranks))
  mpiexec -np ${n_ranks} python3 train.py --distributed \
    --output-dir "results/${output_dir_postfix}/scaling-n${n_ranks}" \
    --n-train $((2 * ${n_ranks})) --n-valid $((2 * ${n_ranks})) --batch-size 1 --n-epochs 8 \
    --conv-size 2 --n-conv-layers 1 --fc1-size 16 --fc2-size 8 \
    --verbose configs/cosmo_local.yaml
done

# Scaling locally with dummy data
for log_n_ranks in $(seq 0 2); do
  n_ranks=$((2**log_n_ranks))
  mpiexec -np ${n_ranks}  python train.py --distributed \
    --output-dir "results/${output_dir_postfix}/scaling-dummy-n${n_ranks}" \
    --n-train $((2 * ${n_ranks})) --n-valid $((2 * ${n_ranks})) --batch-size 1 --n-epochs 8 \
    --conv-size 2 --n-conv-layers 1 --fc1-size 16 --fc2-size 8 \
    --verbose configs/cosmo_dummy_local.yaml
done
