#!/bin/bash

set -ex
timestamp=$(date "+%Y-%m-%d_%H-%M-%S")
output_dir_postfix="${timestamp}_${HOSTNAME}"
data_dir=/root/mlperf/data/cosmoflow/cosmoUniverse_2019_05_4parE_tf
n_ranks=64
n_runs=1
#n_train_per_rank=256
#n_valid_per_rank=64
#n_epochs=5
set +x


for instance in $(seq 1 ${n_runs}); do
  set -x
  sbatch -N ${n_ranks}  scripts/daint/train_sarus.sh  \
      --data-dir ${data_dir} \
      --output-dir "results/mlperf_run/${output_dir_postfix}/${instance}" \
      configs/cosmo.yaml # dry-run: cosmo.yaml -> cosmo_dryrun.yaml
      #--n-train $((${n_train_per_rank} * ${n_ranks})) --n-valid $((${n_valid_per_rank} * ${n_ranks})) --n-epochs ${n_epochs} \
  set +x
done

