#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : transfer_learning.py
#   Author      : ronen halevy 
#   Created date:  8/10/22
#   Description :
#
# ================================================================
import yaml
from tensorflow.keras import Input

from core.parse_model import ParseModel



def copy_weights(model, ref_model, transfer_learning_submodels, weights_file_path):
    for layer in model.layers:
        submodels = list(filter(lambda x: x in layer.name ,transfer_learning_submodels))
        if submodels:
            layer.set_weights(ref_model.get_layer(
                layer.name).get_weights())


def freeze_model(model, freeze_models, freeze_flag=True):
    for layer in model.layers:
        submodels = list(filter(lambda x: x in layer.name, freeze_models))
        if submodels:
            layer.trainable = not freeze_flag

def disable_bn(model, freeze_bn, freeze_flag=True):
    for layer in model.layers:
        submodels = list(filter(lambda x: x in layer.name, freeze_bn))
        if submodels:
            layer.training = not freeze_flag


def do_transfer_learning(model, ref_model, transfer_learning_cfg, input_weights_path):
    if transfer_learning_cfg.get('load_weights') and 'none' not in transfer_learning_cfg['load_weights']:
        copy_weights(model, ref_model, transfer_learning_cfg['load_weights'],
                      input_weights_path)
    if transfer_learning_cfg['freeze_training']:
        freeze_model(model, transfer_learning_cfg['freeze_training'])

    if transfer_learning_cfg['batch_norm_freeze']:
        disable_bn(model, transfer_learning_cfg['batch_norm_freeze'])


if __name__ == '__main__':
    model_config_file = 'config/models/yolov3/model.yaml'
    with open(model_config_file, 'r') as _stream:
        model_config = yaml.safe_load(_stream)
    parse_model = ParseModel()
    nclasses = 80
    inputs = Input(shape=(None, None, 3), name = 'input')
    model = parse_model.build_model(inputs, nclasses, **model_config)


    with open('config/train_config.yaml', 'r') as _stream:
        train_config = yaml.safe_load(_stream)
    transfer_learning_cfg = train_config['transfer_learning_config']
    input_weights_path = train_config['input_weights_path']
    model.load_weights(input_weights_path)

    do_transfer_learning(transfer_learning_cfg, input_weights_path)