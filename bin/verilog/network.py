


import os
from env2 import tf, np, plt
from env2 import global_tf_float_precision

from ase.io import read, write

config = {
    "config_file" : './config.npy',
    "weight_file" : './weight.npy',
    "map_file" : './map.npy',
    "atom_file" : './atoms.xsf',
    "atom_past_file" : './atoms_past.xsf',
    "fpga_model_file" : 'fpga_model.txt',
    "verilog_file": os.path.dirname(os.path.realpath(__file__)) + "/../data/verilog.npy",
    "debug" : {},
    "cf" : {},
    "net" : {},
    "map" : {},
    "atoms" : {},
    "atoms_past" : {},
    "end" : []
}


def set_config(key, value):
    config[key] = value

def get_config(key):
    if key in config.keys():
        return config[key]
    else:
        return None

# Get Data
# =====================================================================

def read_cf():
    cf = get_config('cf')
    if len(cf) == 0:
        cf = np.load(get_config('config_file'),allow_pickle=True)[0]
        set_config('cf', cf)
    return cf

def read_net():
    net = get_config('net')
    if len(net) == 0:
        net = np.load(get_config('weight_file'),allow_pickle=True)[0]
        set_config('net', net)
    return net

def read_map():
    map = get_config('map')
    if len(map) == 0:
        map = np.load(get_config('map_file'),allow_pickle=True)[0]
        set_config('map', map)
    return map

def read_atoms():
    atoms = get_config('atoms')
    if len(atoms) == 0:
        atoms = np.load(get_config('atom_file'),allow_pickle=True)[0]
        set_config('atoms', atoms)
    return atoms

def read_atoms_past():
    atoms = get_config('atoms_past')
    if len(atoms) == 0:
        atoms = np.load(get_config('atom_past_file'),allow_pickle=True)[0]
        set_config('atoms_past', atoms)
    return atoms

def read_data(key):
    dat = np.loadtxt("./out/%s.txt"%key)
    return dat

def get_dic(dic, key):
    keys = key.split(',')
    lst = []
    for key in keys:
        lst.append(dic[key])
    return lst


# Output Data
# =====================================================================

def plot_data(xs, ys, close=True, mark='.', labels=',', fn=''):
    x = xs
    ys = ys if type(ys) == list else [ys]
    labels = labels.split(',')

    ct = 0
    for y in ys:
        ct += 1
        sh = y.shape
        y = np.reshape(y, [sh[0], -1])
        sh = y.shape

        for ii in range(sh[1]):
            plt.plot(x, y[:,ii], mark, label="%d"%ii)

        plt.grid()
        plt.xlabel(labels[0], fontsize=20)
        plt.ylabel(labels[1], fontsize=20)
        plt.title(labels, fontsize=20)
        if fn != '':
            plt.savefig(fn+"-%d.png"%(ct))
            plt.close()
        elif close:
            plt.show()
            plt.close()


# Network
# =====================================================================

def get_sess():
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    return sess

def get_var(names, shape, precision, net=None):
    """
    创建随机变量，和重新加载参数的变量
    """
    if net == None:
        net = read_net()
    tmp_w = tf.get_variable(names[0]+'_XXX',[1])
    name = tmp_w.name.replace(':0','').replace('/','_').replace('_XXX','')
    value = net[name]
    init_w = tf.constant_initializer(value)

    tmp_b = tf.get_variable(names[1]+'_XXX',[1])
    name = tmp_b.name.replace(':0','').replace('/','_').replace('_XXX','')
    value = net[name]
    init_b = tf.constant_initializer(value)

    w = tf.get_variable(names[0], 
                            [shape[0], shape[1]], 
                            precision,
                            init_w, 
                            trainable = False)
    b = tf.get_variable(names[1], 
                        [shape[1]], 
                        precision,
                        init_b, 
                        trainable = False)
    return w, b

def v2s2v (v):
    v2 = []
    s = ""
    for vi in v:
        si = "%16.10f"%vi
        s += si + " "
        v2.append(float(si))
    print(s)
    return v2

def pow_2n(xv):
    v = tf.constant(np.log(1.5) / np.log(2), dtype=global_tf_float_precision)
    eps = 1e-12
    xlog = tf.log(tf.abs(xv) + eps) * (1 / np.log(2.0))
    v2n = tf.pow(2.0, tf.ceil(xlog - v))
    return v2n
