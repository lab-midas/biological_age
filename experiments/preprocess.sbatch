#!/bin/bash
####
#SBATCH --job-name=ukbpreproc
#SBATCH --partition=week
#SBATCH --time 7-0
#SBATCH --nodelist=node-cpu-01
#SBATCH --error=job.%J.%x.err
#SBATCH --output=job.%J.%x.out

cd ~/Documents/nako_ukb_age/preprocess
/opt/conda/rakuest1/envs/nakoukb/bin/python3 dcm2nii.py /mnt/qdata/rawdata/UKBIOBANK/ukbdata/brain/t1/Dicom /mnt/qdata/share/rakuest1/data/UKB/raw/t1_brain