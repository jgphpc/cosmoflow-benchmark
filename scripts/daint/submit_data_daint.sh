#!/bin/bash

set -ex
timestamp=$(date "+%Y-%m-%d_%H-%M-%S")
output_dir_postfix="${timestamp}_${HOSTNAME}"
data_dir=/scratch/snx3000/lukasd/mlperf/data/cosmoflow/cosmoUniverse_2019_05_4parE_tf
log_max_ranks=7
set +x


# Weak scaling on Piz Daint (simulates 128 Cosmoflow nodes)
for log_n_ranks in $(seq 0 ${log_max_ranks}); do
  n_ranks=$((2**log_n_ranks))
  set -x
  sbatch -N ${n_ranks}  scripts/daint/train_daint.sh --data-benchmark  \
      --data-dir ${data_dir} \
      --output-dir "results/data_benchmark/${output_dir_postfix}/gpu-n${n_ranks}" \
      --n-train $((2048 * ${n_ranks})) --n-valid $((512 * ${n_ranks})) --n-epochs 10 \
      configs/cosmo.yaml
  set +x
done

# Strong scaling also of interest(?)


# Reference with dummy data is not expected to be needed as higher epochs can serve for this purpose 
#for log_n_ranks in $(seq 0 0); do
#  n_ranks=$((2**log_n_ranks))
#  set -x
#  sbatch -N ${n_ranks}  scripts/daint/train_daint.sh --data-benchmark \
#      --output-dir "results/data_benchmark/${output_dir_postfix}/gpu-dummy-n${n_ranks}" \
#      --n-train $((2048 * ${n_ranks})) --n-valid $((512 * ${n_ranks})) --n-epochs 5 \
#      configs/cosmo_dummy.yaml
#  set +x
#done
