#!/bin/bash
#SBATCH --job-name=reinvent_job
#SBATCH --output=00_reinventLogs/reinvent_%j.out
#SBATCH --time=28:00:00
#SBATCH --mem=25GB
# --- Cluster-specific: uncomment and edit for your own Slurm site ---
##SBATCH --partition=<your-cpu-partition>
##SBATCH --account=<your-account>
# Create the log directory before first submission: mkdir -p 00_reinventLogs

# Make `conda activate` work in a non-interactive batch shell. Edit the path to
# your conda installation, or replace the two conda lines with your own activation.
# source /path/to/conda/etc/profile.d/conda.sh
conda activate reinvent4

config=$1

# CPU-only run: hide any GPUs so REINVENT/torch fall back to CPU.
export CUDA_VISIBLE_DEVICES=""

reinvent --log-level debug $config

conda deactivate
