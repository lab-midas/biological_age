#!/bin/bash
####
#SBATCH --job-name=nakopresa
#SBATCH --partition=month
#SBATCH --time 31-0
#SBATCH --cpus-per-task=10
#SBATCH --nodelist=node-gpu-04
#SBATCH --error=job.%J.%x.err
#SBATCH --output=job.%J.%x.out

SINGULARITYENV_PYTHONPATH=/home/raecker1/nako_ukb_age singularity exec --bind /mnt --nv /home/raecker1/age.sif python3 /home/raecker1/nako_ukb_age/preprocess/ukbheart.py /mnt/qdata/share/raecker1/ukbdata_70k/sa_heart/raw /mnt/qdata/share/raecker1/ukbdata_70k/interim

#cd ~/nako_ukb_age/preprocess
#/opt/conda/raecker1/envs/nakoukb/bin/python3 dcm2nii.py /mnt/qdata/rawdata/NAKO_706/NAKO_706_MRT/Cine-SSFP-SAX /mnt/qdata/share/raecker1/nako_30k/sa_heart --study sa_heart --cores 16
#/opt/conda/raecker1/envs/nakoukb/bin/python3 dcm2nii.py /mnt/qdata/rawdata/UKBIOBANK/ukbdata_70k/sa_heart/raw_dicom /mnt/qdata/share/raecker1/ukbdata_70k/sa_heart --study sa_heart --cores 16
#/opt/conda/raecker1/envs/nakoukb/bin/python3 ukbheart.py /mnt/qdata/share/raecker1/ukbdata_70k/sa_heart /mnt/qdata/share/raecker1/ukbdata_70k/interim
#module load cuda/11.4
#/opt/conda/rakuest1/envs/nakoukb/bin/python3 ukbheart.py /mnt/qdata/share/rakuest1/data/UKB/raw/sa_heart/raw /mnt/qdata/share/rakuest1/data/UKB/interim
