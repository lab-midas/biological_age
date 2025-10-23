#!/bin/bash
#export PATH="$HOME/.pyenv/bin:$PATH"
#eval "$(pyenv init -)"
#eval "$(pyenv virtualenv-init -)"
# export PRJ=/home/rakuest1/Dropbox/Promotion/Matlab/DeepLearning/nako_ukb_age  # MIDASGPUWorker
export PRJ=/home/rakuest1/Documents/nako_ukb_age  # denbi
export DATA=/mnt/qdata/share/rakuest1/data/
export OUT=/mnt/qdata/share/rakuest1/results/UKB
export PATH=/opt/conda/rakuest1/envs/optox/bin:/opt/miniconda3/condabin:$PATH
export HTTP_PROXY=http://rakuest1:13Pommes27@httpproxy.zit.med.uni-tuebingen.de:88/
export HTTPS_PROXY=http://rakuest1:13Pommes27@httpproxy.zit.med.uni-tuebingen.de:88/
export http_proxy=http://rakuest1:13Pommes27@httpproxy.zit.med.uni-tuebingen.de:88/
export https_proxy=http://rakuest1:13Pommes27@httpproxy.zit.med.uni-tuebingen.de:88/


export CONFIG=$PRJ/config/volume/config_ukb.yaml
cd $PRJ
export PYTHONPATH=$PRJ
# MIDASGPUWorker env: optox, denbi: nakoukb
/opt/conda/rakuest1/envs/optox/bin/python $PRJ/brainage/trainer/train.py dataset.fold=0
/opt/conda/rakuest1/envs/optox/bin/python $PRJ/brainage/trainer/train.py dataset.fold=1
/opt/conda/rakuest1/envs/optox/bin/python $PRJ/brainage/trainer/train.py dataset.fold=2
/opt/conda/rakuest1/envs/optox/bin/python $PRJ/brainage/trainer/train.py dataset.fold=3
/opt/conda/rakuest1/envs/optox/bin/python $PRJ/brainage/trainer/train.py dataset.fold=4