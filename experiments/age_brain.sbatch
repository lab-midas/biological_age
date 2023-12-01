#!/bin/bash
####
#SBATCH --job-name=ukbagebrain
#SBATCH --partition=month
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --time 31-0
#SBATCH --nodelist=node-gpu-04
#SBATCH --error=job.%J.%x.err
#SBATCH --output=job.%J.%x.out

export PRJ=/home/raecker1/nako_ukb_age
export DATA=/mnt/qdata/share/raecker1/data
export OUT=/mnt/qdata/share/raecker1/results/UKB
export PATH=/opt/conda/raecker1/envs/bio_age/bin:/opt/miniconda3/condabin:$PATH


#SINGULARITYENV_PYTHONPATH=/home/rafruem1/nako_ukb_age singularity exec --bind /mnt --nv /home/rafruem1/nako_ukb_age.simg \
#            python3 $PRJ/brainage/trainer/train3d.py --fold=0 --batch=8
module load cuda/11.3
export CONFIG=$PRJ/config/volume/config_ukb_brain_meta.yaml
cd $PRJ
export PYTHONPATH=$PRJ
# MIDASGPUWorker env: optox, denbi: nakoukb
/opt/conda/raecker1/envs/bio_age/bin/python $PRJ/brainage/trainer/train3d.py --config $CONFIG
#/opt/conda/rakuest1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train3d.py dataset.fold=0
#/opt/conda/rakuest1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train3d.py dataset.fold=1
#/opt/conda/rakuest1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train3d.py dataset.fold=2
#/opt/conda/rakuest1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train3d.py dataset.fold=3
#/opt/conda/rakuest1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train3d.py dataset.fold=4