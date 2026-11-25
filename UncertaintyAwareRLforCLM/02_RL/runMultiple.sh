#!/bin/bash
# Batch-submit every RL config (*.json) in a folder.
# Usage:   bash runMultiple.sh <config_folder> [job_script]
# Example: bash runMultiple.sh chemprop runRL.sh
set -e

CONFIG_DIR="${1:?Usage: bash runMultiple.sh <config_folder> [job_script]}"
JOB_SCRIPT="${2:-./runRL.sh}"

for config in "$CONFIG_DIR"/*.json; do
    echo "Submitting $JOB_SCRIPT $config"
    sbatch "$JOB_SCRIPT" "$config"
done
