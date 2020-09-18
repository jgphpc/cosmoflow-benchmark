### Benchmarks for Cosmoflow evaluation on Piz Daint

Please clone this repository, set up the data directory (at `../data` relative to it) and build & move the Cosmoflow Docker image for Piz Daint as described in [README_CSCS_SUBMISSION.md](README_CSCS_SUBMISSION.md). Results from every run will appear under `results/<study-id>` and log-files in `logs/`.


### Run an individual instance of Cosmoflow

An individual run of the benchmark with the following `sbatch` command
```
sbatch -N ${n_ranks}  scripts/daint/train_sarus.sh  \
    --output-dir "results/test/$(date '+%Y-%m-%d_%H-%M-%S')_${HOSTNAME}/" \
    --n-train $((${n_training_samples_per_rank} * ${n_ranks})) --n-valid $((${n_validation_samples_per_rank} * ${n_ranks})) --n-epochs ${n_epochs} \
    configs/cosmo.yaml
```
where `scripts/daint/train_sarus.sh` contains the basic `sbatch` configuration and the command line parameters allow to overwrite configuration supplied in the referenced yaml-file in the `configs` directory. To use the weak scaling study parameters, e.g. set
```
n_training_samples_per_rank=256
n_validation_samples_per_rank=256
n_epochs=16
```
and choose `n_ranks` according to your preference.

### Benchmark studies

Data loading throughput can be measured with 

```
scripts/daint/submit_data_daint.sh       # uses gpu-partition
scripts/daint/submit_data_daint_cpu.sh   # uses mc-partition
```

and evaluated with `jupyter lab notebooks/DataBenchmark.ipynb`. This can be used to do CPU thread pool parameter optimization. Weak scaling can be measured with

```
scripts/daint/submit_scaling_daint.sh       # uses gpu-partition
scripts/daint/submit_scaling_daint_cpu.sh   # uses mc-partition
```

and evaluated with `jupyter lab notebooks/ScalingAnalysisDaint.ipynb`.

