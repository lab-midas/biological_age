#!/bin/bash
####
#SBATCH --job-name=ukbpreproc
#SBATCH --partition=week
#SBATCH --time 7-0
#SBATCH --nodelist=node-gpu-04
#SBATCH --error=job.%J.%x.err
#SBATCH --output=job.%J.%x.out

#export PATH=/opt/conda/raecker1/envs/nakoukb/bin:/opt/miniconda3/condabin:$PATH

#SINGULARITYENV_PYTHONPATH=/home/raecker1/nako_ukb_age singularity exec --bind /mnt --nv /home/raecker1/age.sif python3 /home/raecker1/nako_ukb_age/preprocess/ukbbrain.py /mnt/qdata/share/raecker1/ukbdata_70k/t1_brain/processed /mnt/qdata/share/raecker1/ukbdata_70k/interim/

cd ~/nako_ukb_age/preprocess
#/opt/conda/raecker1/envs/sklearn-env/bin/python3 ukbbrain.py /mnt/qdata/share/raecker1/data/UKB/brain/processed /mnt/qdata/share/raecker1/data/UKB/brain/interim/
#/opt/conda/raecker1/envs/nakoukb/bin/python3 dcm2nii.py /mnt/qdata/share/rafruem1/ukb/60520_newBatch_Refresh/T1Brain/ /mnt/qdata/share/raecker1/ukbdata_70k/t1_brain/
#/opt/conda/rakuest1/envs/nakoukb/bin/python3 dcm2nii.py /mnt/qdata/share/rafruem1/ukb/MRI/raw/T1Brain /mnt/qdata/share/rakuest1/data/UKB/raw/t1_brain
#/opt/conda/raecker1/envs/nakoukb/bin/python3 -c "import nibabel"
/opt/conda/raecker1/envs/nakoukb/bin/python3 ukbbrain.py /mnt/qdata/share/raecker1/ukbdata_70k/t1_brain/processed /mnt/qdata/share/raecker1/ukbdata_70k/interim/
#/opt/conda/rakuest1/envs/nakoukb/bin/python3 dcm2nii.py /mnt/qdata/share/rafruem1/ukb/MRI/raw/ShortAxisHeart /mnt/qdata/share/rakuest1/data/UKB/raw/sa_heart --study sa_heart --cores 16
#/opt/conda/rakuest1/envs/nakoukb/bin/python3 dcm2nii.py /mnt/qdata/share/rafruem1/ukb/MRI/raw/LongAxisHeart /mnt/qdata/share/rakuest1/data/UKB/raw/la_heart --study la_heart --cores 16

