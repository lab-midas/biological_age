#!/bin/bash
####
#SBATCH --job-name=ukbpresa
#SBATCH --partition=week
#SBATCH --time 7-0
#SBATCH --nodelist=node-cpu-02
#SBATCH --error=job.%J.%x.err
#SBATCH --output=job.%J.%x.out

cd ~/Documents/nako_ukb_age/preprocess
#/opt/conda/rakuest1/envs/nakoukb/bin/python3 dcm2nii.py /mnt/qdata/share/rafruem1/ukb/MRI/raw/ShortAxisHeart /mnt/qdata/share/rakuest1/data/UKB/raw/sa_heart --study sa_heart --cores 16
module load cuda/11.4
/opt/conda/rakuest1/envs/nakoukb/bin/python3 ukbheart.py /mnt/qdata/share/rakuest1/data/UKB/raw/sa_heart/raw /mnt/qdata/share/rakuest1/data/UKB/interim