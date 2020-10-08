#!/bin/bash

set -ex

timestamp=$(date "+%Y-%m-%d_%H-%M-%S")
output_dir_postfix="${timestamp}_${HOSTNAME}"
# different data_dir in Sarus vs plain Daint
data_dir=/root/mlperf/data/cosmoflow/cosmoUniverse_2019_05_4parE_tf # /scratch/snx3000/lukasd/mlperf/data/cosmoflow/cosmoUniverse_2019_05_4parE_tf
#log_n_ranks=6
#n_ranks=$(( 2**${log_n_ranks} ))
n_runs=1
set +x

min_log_n_ranks=5
for max_log_n_ranks in $(seq 9 -1 ${min_log_n_ranks}); do
    for log_n_ranks in $(seq ${max_log_n_ranks} -1 ${min_log_n_ranks}); do
        n_ranks=$(( 2**${log_n_ranks} ))
        batch_size=$(( 2**(${max_log_n_ranks}-${log_n_ranks}) ))

#if [ "${n_ranks}" -eq 32 ] && [ "${batch_size}" -eq 8 ]; then
if [ "${batch_size}" -le 8 ]; then
#n_ranks=$(( ${n_ranks}/8 ))

for instance in $(seq 1 ${n_runs}); do
  set -x
  #echo "${n_ranks} ${batch_size}"
  # Replace train_sarus.sh by train_daint.sh to use Daint module
  sbatch -N ${n_ranks}  scripts/daint/train_sarus.sh  \
      --data-dir ${data_dir} \
      --batch-size ${batch_size} --n-epochs 5 \
      --output-dir "results/mlperf_throughput/${output_dir_postfix}/gpu-n${n_ranks}_batch${batch_size}" \
      configs/cosmo.yaml # dry-run: cosmo.yaml -> cosmo_dryrun.yaml
      #--output-dir "results/mlperf_run/${output_dir_postfix}/${instance}"
      #--n-train $((2**(2+10+${log_n_ranks}))) --n-valid $((2**(10+${log_n_ranks}))) --n-epochs 5 \
      #--n-train $((256 * ${n_ranks})) --n-valid $((64 * ${n_ranks})) --n-epochs 3 \
      #--n-train $((${n_train_per_rank} * ${n_ranks})) --n-valid $((${n_valid_per_rank} * ${n_ranks})) --n-epochs ${n_epochs} \
  set +x
done

fi

    done
done
