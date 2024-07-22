#!/bin/bash
####
#SBATCH --job-name=ukbabdpreproc
#SBATCH --partition=week
#SBATCH --time 7-0
#SBATCH --cpus-per-task=10
#SBATCH --nodelist=node-cpu-02
#SBATCH --error=job.%J.%x.err
#SBATCH --output=job.%J.%x.out

SINGULARITYENV_PYTHONPATH=/home/raecker1/nako_ukb_age singularity exec --bind /mnt --nv /home/raecker1/age.sif python3 /home/raecker1/nako_ukb_age/preprocess/extract_ukbb.py --zip_folder /mnt/qdata/share/rafruem1/ukb/60520_newBatch_Refresh/AbdominalMRI/ --nifti_folder /mnt/qdata/share/raecker1/ukbdata_70k/abdominal_MRI/test/

#cd ~/nako_ukb_age/preprocess
#/opt/conda/raecker1/envs/sklearn-env/bin/python3 ukbbrain.py /mnt/qdata/share/raecker1/data/UKB/brain/processed /mnt/qdata/share/raecker1/data/UKB/brain/interim/
#/opt/conda/raecker1/envs/nakoukb/bin/python3 extract_ukbb.py --zip_folder /mnt/qdata/share/rafruem1/ukb/60520_newBatch_Refresh/AbdominalMRI/ --nifti_folder /mnt/qdata/share/raecker1/ukbdata_70k/abdominal_MRI/
#/opt/conda/rakuest1/envs/nakoukb/bin/python3 dcm2nii.py /mnt/qdata/share/rafruem1/ukb/MRI/raw/T1Brain /mnt/qdata/share/rakuest1/data/UKB/raw/t1_brain
#/opt/conda/rakuest1/envs/nakoukb/bin/python3 ukbbrain.py /mnt/qdata/share/rakuest1/data/UKB/raw/t1_brain/processed /mnt/qdata/share/rakuest1/data/UKB/interim/
#/opt/conda/rakuest1/envs/nakoukb/bin/python3 dcm2nii.py /mnt/qdata/share/rafruem1/ukb/MRI/raw/ShortAxisHeart /mnt/qdata/share/rakuest1/data/UKB/raw/sa_heart --study sa_heart --cores 16
#/opt/conda/rakuest1/envs/nakoukb/bin/python3 dcm2nii.py /mnt/qdata/share/rafruem1/ukb/MRI/raw/LongAxisHeart /mnt/qdata/share/rakuest1/data/UKB/raw/la_heart --study la_heart --cores 16
