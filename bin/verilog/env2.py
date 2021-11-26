

import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
import matplotlib.pyplot as plt

global_tf_float_precision = tf.float32
global_np_float_precision = np.float32
global_ener_float_precision = np.float64
global_float_prec = 'float'

import os

def get_module(module_name):
    ext = "so"
    module_path = os.path.dirname(os.path.realpath(__file__)) + "/../train/"
    print("INFO: train path\n", module_path)
    path = module_path  + "{}.{}".format(module_name, ext)
    print("INFO: module path\n", path)
    assert (os.path.isfile (path), "module %s does not exist" % module_name)
    module = tf.load_op_library(path)
    return module

op_module = get_module("tfOp/libdpOp")


intt = np.int64

