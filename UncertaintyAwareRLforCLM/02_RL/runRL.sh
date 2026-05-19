#!/bin/bash
#SBATCH --job-name=00_reinventLogs/reinvent_job
#SBATCH --output=00_reinventLogs/reinvent_%j.out
#SBATCH --time=48:00:00
#SBATCH --partition=medium-gpu
#SBATCH --account=mai
#SBATCH --gres=gpu:a100
#SBATCH --mem=25GB





nvidia-smi

conda activate reinvent4
config=$1


export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
 

reinvent --log-level debug $config

conda deactivate