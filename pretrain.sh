#!/bin/bash

apt-get update
apt-get install openssh-client -y

ssh -o StrictHostKeyChecking=no git@github.com
ssh-agent sh -c 'ssh-add ssh/id_rsa_ml_utils; pip install -r sm_requirements.txt'

python -m spacy download en_core_web_lg
python pretrain.py --config_file ${SM_HPS["config_file"]} ${@:1:$#-2}

mv /opt/ml/code/models /opt/ml/model
