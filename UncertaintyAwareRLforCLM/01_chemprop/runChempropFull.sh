#!/bin/bash
#SBATCH --job-name=chemprop_train
#SBATCH --output=chemprop_%j.out
#SBATCH --time=5:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
# --- Cluster-specific: uncomment and edit for your own Slurm site ---
##SBATCH --partition=<your-gpu-partition>
##SBATCH --account=<your-account>

# Make `conda activate` work in a non-interactive batch shell. Edit the path to
# your conda installation, or replace the two conda lines with your own activation.
# source /path/to/conda/etc/profile.d/conda.sh
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
