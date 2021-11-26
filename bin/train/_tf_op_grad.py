#!/usr/bin/env python3
"""
Gradients for prod force.
"""
import sys


from tensorflow.python.framework import ops

from env import tf
from env import op_module as op_module_dp
from env import op_module2 as op_module_mz
from env import op_grads_module
     
@ops.RegisterGradient("MzProdForceSeA")
def _prod_force_se_a_grad_cc (op, grad):    
    net_grad =  op_grads_module.mz_prod_force_se_a_grad (grad, 
                                                       op.inputs[0], 
                                                       op.inputs[1], 
                                                       op.inputs[2], 
                                                       op.inputs[3], 
                                                       n_a_sel = op.get_attr("n_a_sel"),
                                                       n_r_sel = op.get_attr("n_r_sel"))
    return [net_grad, None, None, None]

@ops.RegisterGradient("MzProdVirialSeA")
def _prod_virial_se_a_grad_cc (op, grad, grad_atom):    
    net_grad =  op_grads_module.mz_prod_virial_se_a_grad (grad, 
                                                        op.inputs[0], 
                                                        op.inputs[1], 
                                                        op.inputs[2], 
                                                        op.inputs[3], 
                                                        op.inputs[4], 
                                                        n_a_sel = op.get_attr("n_a_sel"),
                                                        n_r_sel = op.get_attr("n_r_sel"))
    return [net_grad, None, None, None, None]

@ops.RegisterGradient("Mzscatter")
def _MzscatterGrad(op, grad):
    f = op.inputs[0]
    l = op.inputs[1]
    Ni = op.get_attr("ni")
    grad_f = op_module_mz.mzscatter_grad(f, l, grad, Ni)
    return [grad_f, None]


@ops.RegisterGradient("Mzquantify")
def _MzquantifyGrad(op, grad):
    isround = op.get_attr("isround")
    nbit1 = op.get_attr("nbit1")
    nbit2 = op.get_attr("nbit2")
    nbit3 = op.get_attr("nbit3")
    dx = op_module_mz.mzquantify(grad, isround, nbit2, nbit3, nbit1)
    # print("INFO: Mzquantify")
    # print("dy", grad)
    # print("x", op.inputs[0])
    # print("y", op.outputs[0])
    # print("dx", dx)
    return dx

@ops.RegisterGradient("Mztanh2")
def _Mztanh2Grad(op, grad):
    isround = op.get_attr("isround")
    nbit1 = op.get_attr("nbit1")
    nbit2 = op.get_attr("nbit2")
    nbit3 = op.get_attr("nbit3")
    prec = 2 ** nbit2
    x = op.inputs[0]
    x_abs = tf.abs(x)
    x1 = tf.clip_by_value(x_abs, 0, 2)
    x2 = tf.clip_by_value(x_abs, 0, 4)
    dydx = (132-64*x1-x2) * 0.0078125
    if (nbit2 > -1):
        dydx = dydx + tf.stop_gradient( tf.floor(dydx * prec) / prec - dydx )
    dx = dydx * grad
    if (nbit2 > -1):
        dx = dx + tf.stop_gradient( tf.floor(dx * prec) / prec - dx )
    return dx

@ops.RegisterGradient("Mzmatmul")
def _MzmatmulGrad(op, grad):
    x = op.inputs[0]
    w = op.inputs[1]
    isround = op.get_attr("isround")
    nbit1 = op.get_attr("nbit1")
    nbit2 = op.get_attr("nbit2")
    nbit3 = op.get_attr("nbit3")
    dx = op_module_mz.mzmatmul(grad, tf.transpose(w), isround, nbit2, nbit3, nbit1)
    dw = op_module_mz.mzmatmul(tf.transpose(x), grad, isround, nbit2, nbit3, nbit1)
    return [dx, dw]

@ops.RegisterGradient("Mzmap")
def _MzmapGrad(op, grad):
    x = op.inputs[0]
    w = op.inputs[1]
    w2 = op.inputs[2]
    y = op.outputs[0]
    dydx = op_module_mz.mzmap(x, w2, tf.zeros_like(w2))
    # grad = tf.floor(grad * 2**14) / (2**14)
    dx = tf.reshape(tf.reduce_sum(dydx * grad, axis=1), [-1, 1])
    # dx = tf.floor(dx * 2**14) / (2**14)

    dw = None
    dw2 = None
    # print("INFO: _MzmapGrad")
    # print("x", x)
    # print("y", y)
    # print("dy", grad)
    # print("dydx", dydx)
    # print("dx", dx)
    return [dx, dw, dw2]