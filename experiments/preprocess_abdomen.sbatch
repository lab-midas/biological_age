#!/bin/bash
####
#SBATCH --job-name=nako2nii
#SBATCH --partition=month
#SBATCH --time 31-0
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --nodelist=node-gpu-05
#SBATCH --error=job.%J.%x.err
#SBATCH --output=job.%J.%x.out

cd /home/raecker1/nako_ukb_age/preprocess

/opt/conda/raecker1/envs/nakoukb/bin/python3 dcm2nii.py /mnt/qdata/rawdata/NAKO_706/NAKO_706_MRT/3D_GRE_TRA_W_COMPOSED /mnt/qdata/rawdata/NAKO_706/NAKO_706_nifti/3D_GRE_TRA_W_COMPOSED --nako --dixon -v --cores 16
#/opt/conda/rakuest1/envs/nakoukb/bin/python3 stitching_script.py /mnt/qdata/share/rafruem1/ukb/MRI/raw/abdominalMri /mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/raw
#/opt/conda/rakuest1/envs/nakoukb/bin/python3 ukbabdomen.py /mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/raw/ /mnt/qdata/share/rakuest1/data/UKB/interim/

#cd ~/Documents/nako_ukb_age/preprocess/UKBB-GNC-Abdominal-Segmentation
#/opt/conda/rakuest1/envs/nakoukb/bin/python3 convert2nnunet.py --nifti_folder /mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/raw --nnunet_folder /mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/processed/nnunet --dataset_name ukbb --num_channels 4
#module load cuda/11.3
#export CUDA_VISIBLE_DEVICES=0
#export RESULTS_FOLDER=/home/rakuest1/Documents/nako_ukb_age/preprocess/UKBB-GNC-Abdominal-Segmentation/models/
#/opt/conda/rakuest1/envs/nakoukb/bin/python3 predict.py --nnunet_folder /mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/processed/nnunet --prediction_folder /mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/processed/seg --dataset_name ukbb --num_channels 4
#/opt/conda/rakuest1/envs/nakoukb/bin/python3 convert2original.py --prediction_folder /mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/processed/seg --output_folder /mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/processed/seg_ori

# redo patient segmentations
#/opt/conda/rakuest1/envs/nakoukb/bin/python3 ukbabdomen.py /mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/raw/ /mnt/qdata/share/rakuest1/data/UKB/tmp/abdominal_MRI/nifti
#/opt/conda/rakuest1/envs/nakoukb/bin/python3 convert2nnunet.py --nifti_folder /mnt/qdata/share/rakuest1/data/UKB/tmp/abdominal_MRI/nifti --nnunet_folder /mnt/qdata/share/rakuest1/data/UKB/tmp/abdominal_MRI/nnunet --dataset_name ukbb --num_channels 4
#/opt/conda/rakuest1/envs/nakoukb/bin/python3 predict.py --nnunet_folder /mnt/qdata/share/rakuest1/data/UKB/tmp/abdominal_MRI/nnunet --prediction_folder /mnt/qdata/share/rakuest1/data/UKB/tmp/abdominal_MRI/seg --dataset_name ukbb --num_channels 4
#/opt/conda/rakuest1/envs/nakoukb/bin/python3 convert2original.py --prediction_folder /mnt/qdata/share/rakuest1/data/UKB/tmp/abdominal_MRI/seg --output_folder /mnt/qdata/share/rakuest1/data/UKB/tmp/abdominal_MRI/seg_ori
/opt/conda/raecker1/envs/nakoukb/bin/python3 get_age_from_dicom.py