#!/bin/bash
#SBATCH --job-name=reinvent_job
#SBATCH --output=reinvent_%j.out
#SBATCH --time=5:00:00
#SBATCH --partition=medium-gpu
#SBATCH --gres=gpu:a100
#SBATCH --mem=20GB




nvidia-smi
conda activate chemprop


chemprop train \
    -i ../00_data/egfr/cleaned_egfr.csv\
    --task-type regression-mve\
    --smiles-columns "canonical_smiles"\
    --target-columns "pChEMBL Value"\
    --splits-column split\
    --save-dir ./chemprop_model_mve_egfr_full\
    --ensemble-size 10\
    --epochs 200

conda deactivate

