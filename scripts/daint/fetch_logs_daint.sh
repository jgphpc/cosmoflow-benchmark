#!/bin/bash

# Parameters are derived from folder structure in results directory
# $1 is the experiment type (weak_scaling, data_benchmark, etc.)
# $2 is the timestamp_hostname label

scp -r "daint101:/scratch/snx3000/lukasd/mlperf/cosmoflow-benchmark/logs/{$(logfiles=""; while read -r line; do logfiles+=,$line; done < <(ssh daint101 'cd /scratch/snx3000/lukasd/mlperf/cosmoflow-benchmark/logs && grep -l $2 *'); echo "${logfiles:1}")}" results/$1/$2/logs/
