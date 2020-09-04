#!/bin/bash

mkdir --parents /opt/ml/code/data; cp -v -r /opt/ml/input/data/training/* -t $_

apt-get update -qq
apt-get install openssh-client -qq

ssh -o StrictHostKeyChecking=no git@github.com
ssh-agent sh -c 'ssh-add ssh/id_rsa_ml_utils; pip install -r sm_requirements.txt'

python -m spacy download en_core_web_lg
python -W ignore pretrain.py --config_file ${SM_HP_CONFIG_FILE}

mv /opt/ml/code/models /opt/ml/model
mv /opt/ml/code/results /opt/ml/model
