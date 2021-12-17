#!/usr/bin/env python3

from tensorflow.python.framework import ops
from deepmd.env import op_module
from deepmd.env import tf 

@ops.RegisterGradient("MapNvnmd")
def _MapNvnmdGrad(op, grad):
    x = op.inputs[0]
    w = op.inputs[1]
    w2 = op.inputs[2]
    y = op.outputs[0]
    dydx = op_module.map_nvnmd(x, w2, tf.zeros_like(w2))
    dx = tf.reshape(tf.reduce_sum(dydx * grad, axis=1), [-1, 1])

    dw = None
    dw2 = None
    return [dx, dw, dw2]

