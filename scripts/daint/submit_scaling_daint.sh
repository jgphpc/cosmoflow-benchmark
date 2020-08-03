#!/bin/bash

set -ex
timestamp=$(date "+%Y-%m-%d_%H-%M-%S")
output_dir_postfix="${timestamp}_${HOSTNAME}"


# Scaling on Piz Daint # TODO: go up 512 nodes
for log_n_ranks in $(seq 0 3); do
  n_ranks=$((2**log_n_ranks))
  sbatch -N ${n_ranks}  scripts/train_daint.sh  \
      --output-dir "results/${output_dir_postfix}/scaling-cgpu-n${n_ranks}" \
      --n-train $((256 * ${n_ranks})) --n-valid $((256 * ${n_ranks})) --n-epochs 4 \
      configs/cosmo.yaml
done

# Scaling on Piz Daint with dummy data
for log_n_ranks in $(seq 0 3); do
  n_ranks=$((2**log_n_ranks))
  sbatch -N ${n_ranks}  scripts/train_daint.sh  \
      --output-dir "results/${output_dir_postfix}/scaling-cgpu-dummy-n${n_ranks}" \
      --n-train $((256 * ${n_ranks})) --n-valid $((256 * ${n_ranks})) --n-epochs 4 \
      configs/cosmo_dummy.yaml
done
