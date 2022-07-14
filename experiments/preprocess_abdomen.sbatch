#!/bin/bash
####
#SBATCH --job-name=ukbpreabd
#SBATCH --partition=week
#SBATCH --time 7-0
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=node-gpu-05
#SBATCH --error=job.%J.%x.err
#SBATCH --output=job.%J.%x.out

cd ~/Documents/nako_ukb_age/preprocess
#/opt/conda/rakuest1/envs/nakoukb/bin/python3 stitching_script.py /mnt/qdata/share/rafruem1/ukb/MRI/raw/abdominalMri /mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/raw
#/opt/conda/rakuest1/envs/nakoukb/bin/python3 ukbabdomen.py /mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/raw/ /mnt/qdata/share/rakuest1/data/UKB/interim/
cd ~/Documents/nako_ukb_age/preprocess/UKBB-GNC-Abdominal-Segmentation
#/opt/conda/rakuest1/envs/nakoukb/bin/python3 convert2nnunet.py --nifti_folder /mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/raw --nnunet_folder /mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/processed/nnunet --dataset_name ukbb --num_channels 4
module load cuda/11.3
export CUDA_VISIBLE_DEVICES=0
export RESULTS_FOLDER=/home/rakuest1/Documents/nako_ukb_age/preprocess/UKBB-GNC-Abdominal-Segmentation/models/
/opt/conda/rakuest1/envs/nakoukb/bin/python3 predict.py --nnunet_folder /mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/processed/nnunet --prediction_folder /mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/processed/seg --dataset_name ukbb --num_channels 4
#/opt/conda/rakuest1/envs/nakoukb/bin/python3 convert2original.py --prediction_folder /mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/processed/seg --output_folder /mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/processed/seg_ori