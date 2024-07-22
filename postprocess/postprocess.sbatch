#!/bin/bash
####
#SBATCH --job-name=AT_postprocess
#SBATCH --partition=week
#SBATCH --time 7-0
#SBATCH --nodelist=node-gpu-02
#SBATCH --error=job.%J.%x.err
#SBATCH --output=job.%J.%x.out

cd ~/nako_ukb_age/postprocess

/opt/conda/raecker1/envs/nakoukb/bin/python3 calculate_fat_fraction.py
