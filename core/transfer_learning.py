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
from core.parse_model import ParseModel

def transfer_learning(model, model_config_file, nclasses, submodels_names_prefixes, weights_file_path):
    with open(model_config_file, 'r') as _stream:
        model_config = yaml.safe_load(_stream)
    parse_model = ParseModel()
    ref_model, _ = parse_model.build_model(nclasses, **model_config)

    ref_model.load_weights(weights_file_path)
    for layer in model.layers:
        submodels = list(filter(lambda x: x in layer.name ,submodels_names_prefixes))
        if submodels:
            layer.set_weights(ref_model.get_layer(
                layer.name).get_weights())





def freeze_model(model, submodels_names_prefixes, freeze_flag):
    for layer in model.layers:
        submodels = list(filter(lambda x: x in layer.name, submodels_names_prefixes))
        if submodels:
            layer.trainable = freeze_flag

def disable_bn(model, submodels_names_prefixes, freeze_flag):
    for layer in model.layers:
        submodels = list(filter(lambda x: x in layer.name, submodels_names_prefixes))
        if submodels:
            layer.training = freeze_flag




if __name__ == '__main__':
    model_config_file = 'config/models/yolov3/model.yaml'
    weights_file_path = 'stam.tf'
    with open(model_config_file, 'r') as _stream:
        model_config = yaml.safe_load(_stream)
    parse_model = ParseModel()
    nclasses = 7
    model, _ = parse_model.build_model(nclasses, **model_config)
    model.save_weights(
        weights_file_path)

    submodels_names_prefixes = ['head', 'neck']
    transfer_learning(model, model_config_file, nclasses, submodels_names_prefixes, weights_file_path)
    freeze_flag = False
    freeze_model(model, submodels_names_prefixes, freeze_flag)
    disable_bn(model, submodels_names_prefixes, freeze_flag)
    pass
    print('fff')