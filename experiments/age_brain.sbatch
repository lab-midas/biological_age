#!/bin/bash
####
#SBATCH --job-name=ukbagebrain
#SBATCH --partition=day
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time 1-0
#SBATCH --nodelist=node-gpu-02
#SBATCH --error=job.%J.%x.err
#SBATCH --output=job.%J.%x.out

export PRJ=/home/rafruem1/nako_ukb_age  # denbi
export DATA=/mnt/qdata/share/rakuest1/data
export OUT=/mnt/qdata/share/rafruem1/results/UKB
#export PATH=/opt/conda/rakuest1/envs/merlin/bin:/opt/miniconda3/condabin:$PATH
#export HTTP_PROXY=http://rakuest1:13Pommes28@httpproxy.zit.med.uni-tuebingen.de:88/
#export HTTPS_PROXY=http://rakuest1:13Pommes28@httpproxy.zit.med.uni-tuebingen.de:88/
#export http_proxy=http://rakuest1:13Pommes28@httpproxy.zit.med.uni-tuebingen.de:88/
#export https_proxy=http://rakuest1:13Pommes28@httpproxy.zit.med.uni-tuebingen.de:88/


SINGULARITYENV_PYTHONPATH=/home/rafruem1/nako_ukb_age singularity exec --bind /mnt --nv /home/rafruem1/nako_ukb_age.simg \
            python3 $PRJ/brainage/trainer/train3d.py --fold=0 --batch=8
#module load cuda/11.3
# export CONFIG=$PRJ/config/volume/config_ukb_brain.yaml
#cd $PRJ
#export PYTHONPATH=$PRJ
# MIDASGPUWorker env: optox, denbi: nakoukb
#/opt/conda/rakuest1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train3d.py dataset.fold=0
#/opt/conda/rakuest1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train3d.py dataset.fold=1
#/opt/conda/rakuest1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train3d.py dataset.fold=2
#/opt/conda/rakuest1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train3d.py dataset.fold=3
#/opt/conda/rakuest1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train3d.py dataset.fold=4