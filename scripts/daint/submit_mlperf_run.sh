#!/bin/bash

set -ex
timestamp=$(date "+%Y-%m-%d_%H-%M-%S")
output_dir_postfix="${timestamp}_${HOSTNAME}"
data_dir=/root/mlperf/data/cosmoflow/cosmoUniverse_2019_05_4parE_tf
n_ranks=1
n_runs=10
set +x


for instance in $(seq 1 ${n_runs}); do
  set -x
  sbatch -N ${n_ranks}  scripts/daint/train_sarus.sh  \
      --data-dir ${data_dir} \
      --output-dir "results/mlperf_run/${output_dir_postfix}/${instance}" \
      configs/cosmo_dryrun.yaml # FIXME: cosmo_dryrun.yaml -> cosmo.yaml
  set +x
done

