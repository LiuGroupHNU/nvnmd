
import numpy as np

from deepmd.env import tf
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import op_module

from deepmd.nvnmd.utils.config import nvnmd_cfg
from deepmd.nvnmd.utils.weight import get_constant_initializer
from deepmd.utils.network import variable_summaries

def get_sess():
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    return sess

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

def pow_2n(xv, nbit):
    v = tf.constant(np.log2(1.5), dtype=tf.float32)
    ln2 = 1.0/tf.log(2.0)
    eps = 1e-12
    xlog = tf.log(tf.abs(xv) + eps) * ln2
    dn = tf.ceil(xlog - v)
    v2n = tf.pow(2.0, dn)
    v2n = qf(v2n, nbit)
    return v2n

def regular_pow2n(x, M, nbit):
    xv = tf.cast(x, dtype=tf.float32)
    y = 0
    for ii in range(M):
        s = tf.sign(xv)
        v2n = pow_2n(xv, nbit)
        y = y + s*v2n
        xv = xv - s*v2n
    y = tf.cast(y, dtype=GLOBAL_TF_FLOAT_PRECISION)
    y = x + tf.stop_gradient(y - x)
    return y

# fitting_net
def tanh2(x,nbit=-1,nbit2=-1):
    y = op_module.tanh2_nvnmd(x, 0, nbit, nbit2, -1)
    return y

def one_layer_wb(
    shape, 
    outputs_size, 
    bavg, 
    stddev, 
    precision,
    trainable,
    initial_variables, 
    seed, 
    uniform_seed, 
    name):
    
    if nvnmd_cfg.restore_fitting_net:
        # initializer
        w_initializer = get_constant_initializer(nvnmd_cfg.weight, 'matrix')
        b_initializer = get_constant_initializer(nvnmd_cfg.weight, 'bias')
    else:
        w_initializer  = tf.random_normal_initializer(
                            stddev=stddev / np.sqrt(shape[1] + outputs_size),
                            seed=seed if (seed is None or uniform_seed) else seed + 0)
        b_initializer  = tf.random_normal_initializer(
                            stddev=stddev,
                            mean=bavg,
                            seed=seed if (seed is None or uniform_seed) else seed + 1)
        if initial_variables is not None:
            w_initializer = tf.constant_initializer(initial_variables[name + '/matrix'])
            b_initializer = tf.constant_initializer(initial_variables[name + '/bias'])
    # variable
    w = tf.get_variable('matrix', 
                        [shape[1], outputs_size], 
                        precision,
                        w_initializer, 
                        trainable = trainable)
    variable_summaries(w, 'matrix')
    b = tf.get_variable('bias', 
                        [outputs_size], 
                        precision,
                        b_initializer, 
                        trainable = trainable)
    variable_summaries(b, 'bias')

    return w, b

def one_layer(inputs, 
              outputs_size, 
              activation_fn=tf.nn.tanh, 
              precision = GLOBAL_TF_FLOAT_PRECISION, 
              stddev=1.0,
              bavg=0.0,
              name='linear', 
              reuse=None,
              seed=None, 
              use_timestep = False, 
              trainable = True,
              useBN = False, 
              uniform_seed = False,
              initial_variables = None):
    if activation_fn != None: activation_fn = tanh2 
    with tf.variable_scope(name, reuse=reuse):
        shape = inputs.get_shape().as_list()
        w, b = one_layer_wb(shape, outputs_size, bavg, stddev, precision, trainable, initial_variables, seed, uniform_seed, name)
        if nvnmd_cfg.quantize_fitting_net:
            NUM_WLN2 = nvnmd_cfg.nbit['NUM_WLN2']
            NBIT_DATA_FL = nvnmd_cfg.nbit['NBIT_DATA_FL']
            #
            inputs = qf(inputs, NBIT_DATA_FL)
            w = regular_pow2n(w, NUM_WLN2, NBIT_DATA_FL)
            with tf.variable_scope('wx', reuse=reuse):
                wx = op_module.matmul_nvnmd(inputs, w, 0, NBIT_DATA_FL, NBIT_DATA_FL, -1)
                # wx = op_module.quantize_nvnmd(wx, 0, -1, -1, -1)
            #
            b = qr(b, NBIT_DATA_FL)
            with tf.variable_scope('wxb', reuse=reuse):
                hidden = wx + b
                # hidden = op_module.quantize_nvnmd(hidden, 0, -1, -1, -1)
            #
            with tf.variable_scope('actfun', reuse=reuse):
                if activation_fn != None:
                    y = activation_fn(hidden, NBIT_DATA_FL, NBIT_DATA_FL)
                else:
                    y = hidden + 0
                # y = op_module.quantize_nvnmd(y, 0, -1, -1, -1)
        else:
            hidden = tf.matmul(inputs, w) + b
            y = activation_fn(hidden, -1, -1) if activation_fn != None else hidden 
    return y


