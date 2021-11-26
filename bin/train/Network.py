import numpy as np

import os
from env import tf
from env import op_module2
from RunOptions import global_tf_float_precision

def one_layer(inputs, 
              outputs_size, 
              activation_fn=tf.nn.tanh, 
              precision = global_tf_float_precision, 
              stddev=1.0,
              bavg=0.0,
              name='linear', 
              reuse=None,
              seed=None, 
              use_timestep = False, 
              trainable = True,
              useBN = False,
              cfg = {}):
    
    with tf.variable_scope(name, reuse=reuse):
        shape = inputs.get_shape().as_list()
        w, b = get_var(['matrix','bias'], [shape[1], outputs_size], precision, bavg, stddev, seed, cfg)
        w = regular_pow2n(w, 3) if (cfg['retrain']) else w

        if cfg['quantify']:
            cf = read_cf()
            nbit2 = cf['ox091079086097108081078097078108083089'] if (cfg['quantify_grad']) else -1
            inputs = qf(inputs, cf['ox091079086097108081078097078108083089'])
            wx = op_module2.mzmatmul(inputs, w, 0, cf['ox091079086097108081078097078108083089'], nbit2, -1)
            wx = op_module2.mzquantify(wx, 0, -1, -1, -1)
            b = qr(b, cf['ox091079086097108081078097078108083089'])
            hidden = wx + b
            hidden = op_module2.mzquantify(hidden, 0, -1, -1, -1)
            y = activation_fn(hidden, cf['ox091079086097108081078097078108083089'], nbit2) if activation_fn != None else hidden 
            y = op_module2.mzquantify(y, 0, -1, -1, -1)
        else:
            wx = tf.matmul(inputs, w)
            b = b
            hidden = wx + b
            y = activation_fn(hidden) if activation_fn != None else hidden 

        return y

def get_var(names, shape, precision, bavg, stddev, seed, cf):
    trainable = cf['trainable']
    retrain = cf['retrain']
    if retrain:
        net = read_model()

        tmp_w = tf.get_variable(names[0]+'_XXX',[1])
        name = tmp_w.name.replace(':0','').replace('/','_').replace('_XXX','')
        value = net[name]
        init_w = tf.constant_initializer(value)

        tmp_b = tf.get_variable(names[1]+'_XXX',[1])
        name = tmp_b.name.replace(':0','').replace('/','_').replace('_XXX','')
        value = net[name]
        init_b = tf.constant_initializer(value)
    else:
        init_w = tf.random_normal_initializer(stddev=stddev/np.sqrt(shape[0]+shape[1]), seed = seed)
        init_b = tf.random_normal_initializer(stddev=stddev, mean = bavg, seed = seed)
    w = tf.get_variable(names[0], 
                            [shape[0], shape[1]], 
                            precision,
                            init_w, 
                            trainable = trainable)
    b = tf.get_variable(names[1], 
                        [shape[1]], 
                        precision,
                        init_b, 
                        trainable = trainable)
    return w, b

def read_model():
    if os.path.exists('./old-data/model.npy'):
        net = np.load('./old-data/model.npy', allow_pickle=True)[0]
    elif os.path.exists('./data/model.npy'):
        net = np.load('./data/model.npy', allow_pickle=True)[0]
    else:
        net = None
    return net


def pow_2n(xv):
    cf = read_cf()
    v = tf.constant(np.log(1.5) / np.log(2), dtype=global_tf_float_precision)
    eps = 1e-12
    xlog = tf.log(tf.abs(xv) + eps) * (1 / np.log(2.0))
    v2n = tf.pow(2.0, tf.ceil(xlog - v))
    v2n = qf(v2n, cf['ox091079086097108081078097078108083089'])
    return v2n

def regular_pow2n(x, M):
    xv = x
    y = 0
    for ii in range(M):
        s = tf.sign(xv)
        v2n = pow_2n(xv)
        y = y + s*v2n
        xv = xv - s*v2n
    
    y = x + tf.stop_gradient(y - x)
    return y


def matmul2_qq(a, b, nbit):
    sh_a = a.get_shape().as_list()
    sh_b = b.get_shape().as_list()
    a = tf.reshape(a, [-1, 1, sh_a[1]])
    b = tf.reshape(tf.transpose(b), [1, sh_b[1], sh_b[0]])
    y = a * b
    y = qf(y, nbit)
    y = tf.reduce_sum(y, axis=2)
    return y

def matmul3_qq(a, b, nbit):
    sh_a = a.get_shape().as_list()
    sh_b = b.get_shape().as_list()
    a = tf.reshape(a, [-1, sh_a[1], 1, sh_a[2]])
    b = tf.reshape(tf.transpose(b, [0, 2, 1]), [-1, 1, sh_b[2], sh_b[1]])
    y = a * b
    if nbit == -1:
        y = y 
    else:
        y = qf(y, nbit)
    y = tf.reduce_sum(y, axis=3)
    return y

def read_cf():
    if os.path.exists('./old-data/config.npy'):
        cf = np.load('./old-data/config.npy', allow_pickle=True)[0]
    elif os.path.exists('./data/config.npy'):
        cf = np.load('./data/config.npy', allow_pickle=True)[0]
    else:
        cf = None
    return cf

def qf(x, nbit):
    prec = 2**nbit
    
    y = tf.floor(x * prec) / prec
    y = x + tf.stop_gradient(y - x)
    return y

def qr(x, nbit):
    prec = 2**nbit
    
    y = tf.round(x * prec) / prec
    y = x + tf.stop_gradient(y - x)
    return y

def read_map():
    if os.path.exists('./old-data/map.npy'):
        maps = np.load('./old-data/map.npy', allow_pickle=True)[0]
    elif os.path.exists('./data/map.npy'):
        maps = np.load('./data/map.npy', allow_pickle=True)[0]
    else:
        maps = None
    return maps
