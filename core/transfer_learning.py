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



def transfer_weights(model, ref_model, transfer_learning_submodels, weights_file_path):
    for layer in model.layers:
        submodels = list(filter(lambda x: x in layer.name ,transfer_learning_submodels))
        if submodels:
            layer.set_weights(ref_model.get_layer(
                layer.name).get_weights())


def freeze_model(model, freeze_models_list, freeze_flag=True):
    for layer in model.layers:
        submodels = list(filter(lambda x: x in layer.name, freeze_models_list))
        if submodels:
            layer.trainable = not freeze_flag

def disable_bn(model, freeze_bn_list, freeze_flag=True):
    for layer in model.layers:
        submodels = list(filter(lambda x: x in layer.name, freeze_bn_list))
        if submodels:
            layer.training = not freeze_flag


# def do_transfer_learning(model, ref_model, transfer_learning_config, input_weights_path):
def do_transfer_learning(model, model_config, transfer_learning_config, input_weights_path):
    transfer_list = transfer_learning_config['transfer_list']
    parse_model = ParseModel()
    inputs = Input(shape=(None, None, 3))
    complete_transfer_list = ['backbone', 'neck'] if 'neck' in transfer_list else ['backbone']
    sub_models_configs = model_config['sub_models_configs']
    # note: ref_model output stage is according to the stransferred sub-models. Head is not transfered anyway (otherwise, load_weights is used):
    ref_model = parse_model.build_model(inputs, sub_models_configs, output_stage=complete_transfer_list[-1])
    ref_model.load_weights(input_weights_path).expect_partial()

    transfer_weights(model, ref_model, complete_transfer_list,
                      input_weights_path)
    if transfer_learning_config.get('freeze_train_list'):
        freeze_model(model, transfer_learning_config['freeze_train_list'])

    if transfer_learning_config.get('batch_norm_freeze_list'):
        disable_bn(model, transfer_learning_config['batch_norm_freeze_list'])


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
    transfer_learning_config = train_config['transfer_learning_config']
    input_weights_path = train_config['input_weights_path']
    model.load_weights(input_weights_path)

    do_transfer_learning(transfer_learning_config, input_weights_path)