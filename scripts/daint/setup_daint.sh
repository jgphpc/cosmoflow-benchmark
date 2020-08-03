# Source this script to setup the runtime environment on Daint
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load daint-gpu
module load Horovod/0.16.4-CrayGNU-19.10-tf-1.14.0

# Environment variables needed by the NCCL backend
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=ipogif0
export NCCL_IB_CUDA_SUPPORT=1

