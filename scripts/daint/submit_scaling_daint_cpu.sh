#!/bin/bash

set -ex
timestamp=$(date "+%Y-%m-%d_%H-%M-%S")
output_dir_postfix="${timestamp}_${HOSTNAME}"
data_dir=/root/mlperf/data/cosmoflow/cosmoUniverse_2019_05_4parE_tf #/scratch/snx3000/lukasd/mlperf/data/cosmoflow/cosmoUniverse_2019_05_4parE_tf
log_max_ranks=6
set +x


# Scaling on Piz Daint # TODO: go up 512 nodes
for log_n_ranks in $(seq 0 ${log_max_ranks}); do
  n_ranks=$((2**log_n_ranks))
  set -x
  sbatch -N ${n_ranks}  scripts/daint/train_sarus_cpu.sh  \
      --data-dir ${data_dir} \
      --output-dir "results/${output_dir_postfix}/scaling-cpu-n${n_ranks}" \
      --n-train $((32 * ${n_ranks})) --n-valid $((32 * ${n_ranks})) --n-epochs 4 \
      configs/cosmo.yaml
  set +x
done

# Scaling on Piz Daint with dummy data
for log_n_ranks in $(seq 0 ${log_max_ranks}); do
  n_ranks=$((2**log_n_ranks))
  set -x
  sbatch -N ${n_ranks}  scripts/daint/train_sarus_cpu.sh  \
      --output-dir "results/${output_dir_postfix}/scaling-cpu-dummy-n${n_ranks}" \
      --n-train $((32 * ${n_ranks})) --n-valid $((32 * ${n_ranks})) --n-epochs 4 \
      configs/cosmo_dummy.yaml
  set +x
done
