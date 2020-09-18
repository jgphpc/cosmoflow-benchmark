### CSCS Cosmoflow submission

The ML-Perf submission results can be obtained with the docker image 

```
docker build -f builds/Dockerfile.gpu_daint -t cosmoflow_gpu_daint
```

which is built outside of Piz Daint and saved with `docker save cosmoflow_gpu_daint -o cosmoflow_gpu_daint.tar`. To make it available to the container runtime [`Sarus`](https://link.springer.com/chapter/10.1007/978-3-030-34356-9_5), it is copied to Piz Daint and loaded with `sarus load  cosmoflow_gpu_daint.tar cosmoflow_gpu_daint`.

The training data for cosmoflow is expected to reside under `../data/cosmoflow` w.r.t. this repository. To avoid multiple data copies on the same machine when testing different versions of the Cosmoflow code, use e.g. symbolic links to the `data` directory (on Piz Daint to `/scratch/snx3000/lukasd/mlperf/data`).

The submission results can then be obtained with `scripts/daint/submit_mlperf_run.sh`, i.e.

```
sbatch -N 1 scripts/daint/train_sarus.sh --data-dir /root/mlperf/data/cosmoflow/cosmoUniverse_2019_05_4parE_tf --output-dir results/mlperf_dryrun/<XX> configs/cosmo_dryrun.yaml
```
where `<XX>` is a timestamp/instance-label.
