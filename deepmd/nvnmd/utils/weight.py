

import numpy as np 

from deepmd.env import tf
from deepmd.nvnmd.utils.fio import FioHead

def get_weight(weights, key):
    if key in weights.keys():
        return weights[key]
    else:
        head = FioHead().warning()
        print(f"{head}: There is not {key} in weights.")
        return None 

def xp(k1, b1, k2, b2):
    """: calculate the cross point of two lines
    line1: y = k1 * x + b1
    line2: y = k2 * x + b2
    """
    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
    return x, y

def get_r2u_u2r_param(rcut: float):
    rc2 = rcut**2
    th =  np.log(1.5) / np.log(2)

    ln2_k1 = np.ceil(np.log(rc2/4) / np.log(2) -th)
    ln2_k2 = np.ceil(np.log(rc2*1) / np.log(2) -th)
    ln2_k3 = np.ceil(np.log(rc2*4) / np.log(2) -th)

    k1 = np.power(2, -ln2_k1)
    k2 = np.power(2, -ln2_k2)
    k3 = np.power(2, -ln2_k3)
    # b
    b1 = 0
    b2 = 0 #?
    b3 = 1.0 - rc2 * k3
    # xp
    x13, y13 = xp(k1, b1, k2, b2)
    b2 = y13 - x13 * k3
    while(True):
        x12, y12 = xp(k1, b1, k2, b2)
        x23, y23 = xp(k2, b2, k3, b3)
        dx = (x23 - x12) - rc2/2
        if (np.abs(dx) < 1e-6):
            break
        else:
            b2 += 0.01 * dx
    xps = np.array([[x12, y12], [x23, y23]])
    ks = np.array([k1, k2, k3])
    bs = np.array([b1, b2, b3])

    #-u2r-
    ks2 = 1 / (ks * rc2)
    xps2 = np.array([[y12, x12/rc2], [y23, x23/rc2]])
    b2_1 = 0
    b2_2 = xps2[0,1] - xps2[0,0] * ks2[1]
    b2_3 = 1.0 - 1.0 * ks2[2]
    bs2 = [b2_1, b2_2, b2_3]
    return xps, ks, bs, xps2, ks2, bs2

def get_normalize(weights: dict):
    key = f"descrpt_attr.t_avg"
    avg = get_weight(weights,key)
    key = f"descrpt_attr.t_std"
    std = get_weight(weights,key)
    return avg, std 

def get_filter_weight(weights: dict, spe_i: int, spe_j: int, layer_l: int):
    """:
    spe_i(int): 0~ntype-1
    spe_j(int): 0~ntype-1
    layer_l: 1~nlayer
    """
    # key = f"filter_type_{spe_i}.matrix_{layer_l}_{spe_j}" # type_one_side = false
    key = f"filter_type_all.matrix_{layer_l}_{spe_j}" # type_one_side = true
    weight = get_weight(weights,key)
    # key = f"filter_type_{spe_i}.bias_{layer_l}_{spe_j}" # type_one_side = false
    key = f"filter_type_all.bias_{layer_l}_{spe_j}" # type_one_side = true
    bias = get_weight(weights,key)
    return weight, bias 

def get_fitnet_weight(weights: dict, spe_i: int, layer_l: int, nlayer: int=10):
    """:
    spe_i(int): 0~ntype-1
    layer_l(int): 0~nlayer-1
    """
    if layer_l == nlayer - 1 :
        key = f"final_layer_type_{spe_i}.matrix"
        weight = get_weight(weights,key)
        key = f"final_layer_type_{spe_i}.bias"
        bias = get_weight(weights,key)
    else:
        key = f"layer_{layer_l}_type_{spe_i}.matrix"
        weight = get_weight(weights,key)
        key = f"layer_{layer_l}_type_{spe_i}.bias"
        bias = get_weight(weights,key)

    return weight, bias 

def get_constant_initializer(weights, name):
    scope = tf.get_variable_scope().name
    name = scope + '.' + name
    value = get_weight(weights,name)
    return tf.constant_initializer(value)



