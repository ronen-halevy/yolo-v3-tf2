# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : convert_to_tfjs.py
#   Author      : ronen halevy 
#   Created date:  12/1/22
#   Description :
#   This script converts a tf model to a tfjs model. Note that main part of the script is a patch explained inline
# ================================================================

import os
model_in = '/tmp/saved_modeln'
model_tfjs_out = '/tmp/tfjs_modeln'
model_json_file = f'{model_tfjs_out}/model.json'
js_model_dir = '~/develope/tfjs/yolov3-tfjs/models/test_temp/'

# 1. Convert model to tfjs:
###########################
os.system(f"tensorflowjs_converter --input_format=keras_saved_model {model_in}  {model_tfjs_out}")
# Next - here's a patch: Replace "L2" by "L1L2" in model.json. Otherwise, upon loading converted model will fail,with
# an error:
# Unhandled Rejection (Error): Unknown regularizer: L2. This may be due to one of the following reasons:
# The regularizer is defined in Python, in which case it needs to be ported to TensorFlow.js or your JavaScript code.
# The custom regularizer is defined in JavaScript, but is not registered properly with tf.serialization.registerClass().

# Explaination:
#  Model.json expects L2 regularization class - (though here we set no regularization values). However no L2 regularization
# class is pre-implemented in tfjs. So either iplement one, or use L1L2 regularization class which is implemented.
# Taking 2nd solution,  references to "L2" clss name r in model.json are replaced by "L1L2".
##########################################################
fin = open(model_json_file, "rt")
data = fin.read()
data = data.replace('"L2"', '"L1L2"')
fin.close()
fin = open(model_json_file, "wt")
fin.write(data)
fin.close()

# 3. cp file to working dir:
os.system(f"cp {model_tfjs_out}/* {js_model_dir}")

print('Done!!')