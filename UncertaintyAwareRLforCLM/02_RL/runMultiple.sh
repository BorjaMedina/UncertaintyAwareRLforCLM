#!/bin/bash

JOB_SCRIPT="./runRLcpu.sh"
configs= 

for i in $configs; do
    echo "Submitting $JOB_SCRIPT run #$i"
    sbatch "$JOB_SCRIPT" "$i"
    
done