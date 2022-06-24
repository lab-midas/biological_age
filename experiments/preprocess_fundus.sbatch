#!/bin/bash
####
#SBATCH --job-name=ukbfundus
#SBATCH --partition=week
#SBATCH --time 7-0
#SBATCH --nodelist=node-cpu-02
#SBATCH --error=job.%J.%x.err
#SBATCH --output=job.%J.%x.out

cd ~/Documents/nako_ukb_age/preprocess
/opt/conda/rakuest1/envs/nakoukb/bin/python3 ukbfundus.py /mnt/qdata/share/rafruem1/ukb/MRI/raw/NewData/ /mnt/qdata/share/rakuest1/data/UKB/interim