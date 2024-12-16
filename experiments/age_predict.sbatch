#!/bin/bash
####
#SBATCH --job-name=ukbagepredict
#SBATCH --partition=week
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time 7-0
#SBATCH --nodelist=node-gpu-04
#SBATCH --error=job.%J.%x.err
#SBATCH --output=job.%J.%x.out

export PRJ=/home/raecker1/nako_ukb_age  # denbi
export CKPT=/home/raecker1/nako_ukb_age
export DATA=/mnt/qdata/share/raecker1/ukbdata_70k
export OUT=/home/raecker1/nako_ukb_age/results/ukb_age_mainly_healthy_train_set_70k
export PATH=/opt/conda/raecker1/envs/nakoukb/bin:/opt/miniconda3/condabin:$PATH

module load cuda/11.3
cd $PRJ
export PYTHONPATH=$PRJ
# MIDASGPUWorker env: optox, denbi: nakoukb
#/opt/conda/raecker1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train3d.py --config $PRJ/config/volume/config_ukb_brain.yaml --predict $PRJ/config/volume/config_predict_healthy.yaml
#/opt/conda/raecker1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train3d.py --config $PRJ/config/volume/config_ukb_brain_meta.yaml --predict $PRJ/config/volume/config_predict.yaml

#/opt/conda/raecker1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train3d.py --config $PRJ/config/volume/config_ukb_fundus.yaml --predict $PRJ/config/volume/config_predict.yaml
#/opt/conda/raecker1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train3d.py --config $PRJ/config/volume/config_ukb_fundus_meta.yaml --predict $PRJ/config/volume/config_predict.yaml

/opt/conda/raecker1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train3d.py --config $PRJ/config/volume/config_ukb_heart.yaml --predict $PRJ/config/volume/config_predict_healthy.yaml
#/opt/conda/raecker1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train3d.py --config $PRJ/config/volume/config_ukb_heart_meta.yaml --predict $PRJ/config/volume/config_predict.yaml

#/opt/conda/raecker1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train3d.py --config $PRJ/config/volume/config_ukb_liver.yaml --predict $PRJ/config/volume/config_predict.yaml
#/opt/conda/raecker1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train3d.py --config $PRJ/config/volume/config_ukb_liver_meta.yaml --predict $PRJ/config/volume/config_predict.yaml

#/opt/conda/raecker1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train3d.py --config $PRJ/config/volume/config_ukb_kidney.yaml --predict $PRJ/config/volume/config_predict.yaml
#/opt/conda/raecker1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train3d.py --config $PRJ/config/volume/config_ukb_kidney_meta.yaml --predict $PRJ/config/volume/config_predict.yaml

#/opt/conda/raecker1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train3d.py --config $PRJ/config/volume/config_ukb_pancreas.yaml --predict $PRJ/config/volume/config_predict.yaml
#/opt/conda/raecker1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train3d.py --config $PRJ/config/volume/config_ukb_pancreas_meta.yaml --predict $PRJ/config/volume/config_predict.yaml

#/opt/conda/raecker1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train3d.py --config $PRJ/config/volume/config_ukb_spleen.yaml --predict $PRJ/config/volume/config_predict.yaml
#/opt/conda/raecker1/envs/nakoukb/bin/python $PRJ/brainage/trainer/train3d.py --config $PRJ/config/volume/config_ukb_spleen_meta.yaml --predict $PRJ/config/volume/config_predict.yaml
