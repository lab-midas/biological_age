#!/bin/bash
####
#SBATCH --job-name=ukbagespleenmeta
#SBATCH --partition=month
#SBATCH --gpus=1
#SBATCH --time 31-0
#SBATCH --cpus-per-task=24
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodelist=node-gpu-04
#SBATCH --error=job.%J.%x.err
#SBATCH --output=job.%J.%x.out

export PRJ=/home/rakuest1/Documents/nako_ukb_age  # denbi
export DATA=/mnt/qdata/share/rakuest1/data
export OUT=/mnt/qdata/share/rakuest1/results/UKB
export PATH=/opt/conda/rakuest1/envs/nakoukb/bin:/opt/miniconda3/condabin:$PATH

module load cuda/11.3
export CONFIG=$PRJ/config/volume/config_ukb_spleen_meta.yaml
cd $PRJ
export PYTHONPATH=$PRJ
# MIDASGPUWorker env: optox, denbi: nakoukb
/opt/conda/rakuest1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train.py --config $CONFIG
#/opt/conda/rakuest1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train.py dataset.fold=0
#/opt/conda/rakuest1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train.py dataset.fold=1
#/opt/conda/rakuest1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train.py dataset.fold=2
#/opt/conda/rakuest1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train.py dataset.fold=3
#/opt/conda/rakuest1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train.py dataset.fold=4