
import os
import sys
import time
import json
import numpy as np

from RunOptions import global_tf_float_precision, global_ener_float_precision
from Fitting import EnerFitting
from DescrptSeA import DescrptSeA
from Model import Model

from common import j_must_have

from env import tf
from env import default_tf_session_config
from env import get_this_path

from ase.io import read
from ase import Atom

sys.path.append(get_this_path()+'../verilog')
import code_vcs
import network


# config
# =====================================================================

def get_value(dic, key):
    key = key if type(key) == list else [key]

    dic_c = dic
    pars = key[0].split('.')
    for par in pars:
        if par in dic.keys():
            dic = dic[par]
        else:
            dic = None
            break
    if (dic == None) and len(key) > 1:
        return get_value(dic_c, key[1:])
    else:
        return dic

def read_cf():
    return network.read_cf()

def read_map():
    return network.read_map()

def set_cf(cf):
    set_config('cf', cf)

def set_config(key, value):
    network.set_config(key, value)

def get_config(key):
    return network.get_config(key)

# debug
# =====================================================================

class Debug():

    def __init__(self, json_fn):
        fr = open(json_fn, 'r')
        self.jdata = json.load(fr)
        fr.close()

    def build_param(self, jdata):
        self.symbol = get_value(jdata, 'model.type_map')
        self.ntypes = len(self.symbol)

    def build_model(self, jdata):
        self.build_param(jdata)

        model_param = j_must_have(jdata, 'model')
        descrpt_param = j_must_have(model_param, 'descriptor')
        fitting_param = j_must_have(model_param, 'fitting_net')

        self.descrpt = DescrptSeA(descrpt_param)
        self.fitting = EnerFitting(fitting_param, self.descrpt)
        self.model = Model(model_param, self.descrpt, self.fitting)
        # self.model.data_stat(data) #由于map中已包含求avg和std，因此不用此步骤

        self.place_holders = {}
        data_dict = 'box,coord,energy,force'.split(',')
        for kk in data_dict:
            prec = global_tf_float_precision
            self.place_holders[kk] = tf.placeholder(prec, [None], name = 't_' + kk)

        self.place_holders['type']      = tf.placeholder(tf.int32,   [None], name='t_type')
        self.place_holders['natoms_vec']        = tf.placeholder(tf.int32,   [self.ntypes+2], name='t_natoms_vec')
        self.place_holders['default_mesh']      = tf.placeholder(tf.int32,   [None], name='t_default_mesh')
        self.place_holders['is_training']       = tf.placeholder(tf.bool)
        self.model_pred\
            = self.model.build (self.place_holders['coord'], 
                                self.place_holders['type'], 
                                self.place_holders['natoms_vec'], 
                                self.place_holders['box'], 
                                self.place_holders['default_mesh'],
                                self.place_holders,
                                suffix = "", 
                                reuse = False)

# get data
# =====================================================================

def atm_num2spe(atoms):
    """
    (Li,Ge,P,S)->(0,1,2,3)
    """
    cf = read_cf()
    ntype = cf['ox123129134125114112118123114']
    symbol = cf['ox128134122111124121']
    spe = atoms.get_atomic_numbers()
    types = [Atom(symbol[tt]).number for tt in range(ntype)]

    type_dat = []
    for ii in range(len(spe)):
        for tt in range(ntype):
            if spe[ii]==types[tt]:
                type_dat.append(tt)
                break
    type_dat = np.int32(np.array(type_dat))
    return type_dat


def get_data(atoms, dic_ph):
    """The atoms must be resorted by atomic number
    """
    cf = read_cf()
    ntype = cf['ox123129134125114112118123114']
    ntypex = cf['ox123129134125114133']
    symbol = cf['ox128134122111124121']
    natom = len(atoms)

    spe = atoms.get_atomic_numbers()
    types = [Atom(symbol[tt]).number for tt in range(ntype)]

    natoms_vec_dat = [natom, natom]
    for tt in range(ntype):
        n = np.sum(spe==types[tt])
        natoms_vec_dat.append(n)
    natoms_vec_dat = np.int32(np.array(natoms_vec_dat))

    # print("INFO: please resort the atom")
    natoms_vec_dat = [natom, natom]
    type_dat = []
    for tt in range(ntype):
        n = np.sum(spe==types[tt])
        natoms_vec_dat.append(n)
        type_dat.extend([tt]*n)
    natoms_vec_dat = np.int32(np.array(natoms_vec_dat))
    type_dat = np.int32(np.array(type_dat))

    # print("INFO: natoms_vec \n")
    print(natoms_vec_dat)

    coord_dat = np.reshape( atoms.get_positions(), [-1])
    box_dat = np.reshape( atoms.get_cell()+0.0, [-1])
    mesh_dat = np.int32(np.array([0, 0, 0, 2, 2, 2]))
    feed_dic = {
        dic_ph['coord']:coord_dat,
        dic_ph['box']:box_dat,
        dic_ph['type']:type_dat,
        dic_ph['natoms_vec']:natoms_vec_dat,
        dic_ph['default_mesh']:mesh_dat
        }
    return feed_dic


def get_tensor_name(key, cf={}):
    """
    # fea
    x: filter_type_0/Reshape_1:0
    y: filter_type_0/Mzmap:0
    dydx: gradients/filter_type_0/Mzmap_grad/Mzmap:0
    dx: gradients/filter_type_0/Mzmap_grad/Reshape:0
    ## fit
    y:layer_0_type_1/Mzquantify:0
    dx:gradients/layer_0_type_0/Mzquantify_grad/Mzquantify:0
    """
    "fea.0.0.g.dx"
    def find_list(s, ss):
        ss = ss.split(',')
        ivar = -1
        n = len(ss)
        for ii in range(n):
            if s == ss[ii]:
                ivar = ii
        return ivar

    pars = key.split('.')
    
    if pars[0] == 'fea':
        net, tt, tt2, var, xy = pars
        tt = int(tt)
        tt2 = int(tt2)
        stt2 = ['','_1','_2','_3'][tt2]
        # tt
        key_tt = "filter_type_%d"%tt 
        # ii
        ivar = find_list(var, 's,sr,g')
        ixy = find_list(xy, 'x,y,dy,dydx,dx')
        n = 3
        # format
        key_x = "%s/Reshape_%d:0"%(key_tt, 1+5*tt2)
        key_y = "%s/Mzmap_%d:0"%(key_tt, tt2*n+ivar)
        key_dy_s = "gradients/%s/s%s/Mzquantify_grad/Mzquantify:0"%(key_tt, stt2)
        key_dy_sr = "gradients/%s/sr%s/Mzquantify_grad/Mzquantify:0"%(key_tt, stt2)
        key_dy_g = "gradients/%s/g%s/Mzquantify_grad/Mzquantify:0"%(key_tt, stt2)
        key_dys = [key_dy_s, key_dy_sr, key_dy_g]
        key_dy = key_dys[ivar]
        key_dydx = "gradients/%s/Mzmap_%d_grad/Mzmap:0"%(key_tt, tt2*n+ivar)
        key_dx = "gradients/%s/Mzmap_%d_grad/Reshape:0"%(key_tt, tt2*n+ivar)
        keys = [key_x, key_y, key_dy, key_dydx, key_dx]
        keys = [key.replace('Mzmap_0', 'Mzmap') for key in keys]
        key2 = keys[ixy]
    
    if pars[0] == 'fit':
        net, tt, ll, var, xy = pars
        tt = int(tt)
        ll = int(ll)
        # tt, ll
        if cf['final_layer']:
            key_tt = "final_layer_type_%d"%(tt)
        else:
            key_tt = "layer_%d_type_%d"%(ll, tt)
        # ii
        ivar = find_list(var, 'wx,tanh,y')
        ixy = find_list(xy, 'y,dx')
        # format
        key_y = "%s/Mzquantify_%d:0"%(key_tt, ivar)
        key_dx = "gradients/%s/Mzquantify_%d_grad/Mzquantify:0"%(key_tt, ivar)
        keys = [key_y, key_dx]
        keys = [key.replace('Mzquantify_0', 'Mzquantify') for key in keys]
        key2 = keys[ixy]
    
    return key2

def resort_spe(atoms):
    """ resort the order by atomic number
    spe_new = spe[idx]
    spe = spe_new[idx2]
    """
    cf = read_cf()
    Na = len(atoms)
    ntype = cf['ox123129134125114112118123114']

    spe = atoms.get_atomic_numbers()
    spe = atm_num2spe(atoms)
    idx = np.zeros(Na, dtype=np.int32)
    idx2 = np.zeros(Na, dtype=np.int32)
    ct = 0
    for tt in range(ntype):
        for ii in range(Na):
            if spe[ii] == tt:
                idx[ct] = ii 
                idx2[ii] = ct
                ct += 1
    
    return idx, idx2

def sort_by_spe(arr, idx, cf, init_value=0):
    arr = arr[idx]
    return arr

def sort_by_spe2(arr, idx, cf, init_value=0):
    Na = cf['ox091110']
    NI = cf['ox091086']

    arr = np.reshape(arr, [Na, NI])

    for ii in range(Na):
        for jj in range(NI):
            v = int(arr[ii,jj])
            if v == -1:
                break
            v = idx[v]
            arr[ii, jj] = v
    return arr

def resort_lst(lst, cfl):
    """
    重排lst， 由于deepmd的排列lst的方式是按相邻元素种类分成几段的
    """
    Na = cfl['Na']
    NI = cfl['NI']
    NI2 = cfl['NI2']

    lst = np.reshape(lst, [Na, -1])
    idx = np.zeros([Na, NI], dtype=np.int32) - 1

    for ii in range(Na):
        tt = 0
        for jj in range(NI2):
            if lst[ii, jj] != -1:
                idx[ii, tt] = jj
                tt += 1
    
    return idx


def sort_by_lst(arr, idx, cfl, init_value=0):
    """
    使用idx去索引arr，得到新的矩阵arr2
    """
    Na = cfl['Na']
    NI = cfl['NI']
    NI2 = cfl['NI2']

    arr = np.reshape(arr, [Na, NI2, -1])
    arr2 = np.zeros([Na, NI, arr.shape[2]]) + init_value

    for ii in range(Na):
        for jj in range(NI):
            jj2 = idx[ii,jj]
            if jj2 != -1:
                arr2[ii, jj] = arr[ii, jj2]
    return arr2

# test
# =====================================================================

def init_net(json_fn):
    ## 构建网络
    debug = Debug(json_fn)
    debug.build_model(debug.jdata)
    set_config('debug', debug)

    ## sess
    sess = tf.Session(config=default_tf_session_config)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    return debug, sess

def init_atom(atom_fn, atoms=None):
    if atoms == None:
        ## 读入分子系统
        atoms = read(atom_fn)
        idx_spe, idx_spe2 = resort_spe(atoms)
        np.savetxt('idx.txt', idx_spe2, fmt="%d")
        # 重排原子顺序
        atoms = atoms[idx_spe]
    # 获取数据
    debug = get_config('debug')
    feed_dict = get_data(atoms, debug.place_holders)
    return atoms, feed_dict



def init(json_fn, atom_fn):
    debug, sess = init_net(json_fn)
    atoms, feed_dict = init_atom(atom_fn)

    ## cf
    cf = read_cf()
    cf['ox091110'] = len(atoms)

    map = read_map()
    cf.update(map)
    set_cf(cf)

    return debug, sess, atoms, feed_dict, cf

def get_tensor_namelist():
    cf = read_cf()
    ntype = cf['ox123129134125114112118123114']
    ntypex = cf['ox123129134125114133']
    nlayer_fit = cf['ox123121110134114127108115118129']

    # get name
    nameList = [] # 便于标记使用的别名
    nameList2 = [] # tensor的名字
    ## descrpt
    ss = ''
    ss = 't_coord,t_type,t_natoms_vec,t_box,t_default_mesh,'
    ss += 'o_energy,o_force,o_virial,o_atom_energy,o_atom_virial,'
    ss += 'o_rmat,o_rmat_deriv,o_rij,o_nlist,'
    ss += 'o_descriptor,'
    ss = ss.rstrip(',')
    ss = ss.split(',')
    nameList.extend(ss)
    ss = [si+':0' for si in ss]
    nameList2.extend(ss)

    ## S, SR, G
    print("INFO: Add S, SR, G")
    for tt in range(ntypex):
        for tt2 in range(ntype):
            for var in 's,sr,g'.split(','):
                for xy in 'x,y,dy,dydx,dx'.split(','):
                    name = "fea.%d.%d.%s.%s"%(tt, tt2, var, xy)
                    name2 = get_tensor_name(name)
                    nameList.append(name)
                    nameList2.append(name2)
    
    ## GR, DGR, R4
    print("INFO: Add GR, R4")
    for tt in range(ntypex):
        name = "fea.%d.gr"%(tt)
        name2 = "filter_type_%d/gr/Mzquantify:0"%(tt)
        nameList.append(name)
        nameList2.append(name2)

        name = "fea.%d.dgr"%(tt)
        name2 = "gradients/filter_type_%d/gr/Mzquantify_grad/Mzquantify:0"%(tt)
        nameList.append(name)
        nameList2.append(name2)

        name = "fea.%d.d"%(tt)
        name2 = "filter_type_%d/d/Mzquantify:0"%(tt)
        nameList.append(name)
        nameList2.append(name2)

        name = "fea.%d.dd"%(tt)
        name2 = "gradients/filter_type_%d/d/Mzquantify_grad/Mzquantify:0"%(tt)
        nameList.append(name)
        nameList2.append(name2)

        name = "fea.%d.r4.y"%(tt)
        name2 = "filter_type_%d/r4/Mzquantify:0"%(tt)
        nameList.append(name)
        nameList2.append(name2)

        name = "fea.%d.r4.dx"%(tt)
        name2 = "gradients/filter_type_%d/r4/Mzquantify_grad/Mzquantify:0"%(tt)
        nameList.append(name)
        nameList2.append(name2)
    
    ## FIT
    print("INFO: Add FIT")
    for tt in range(ntype):
        for ll in range(nlayer_fit):
            for var in 'wx,tanh,y'.split(','):
                for xy in 'y,dx'.split(','):
                    name = "fit.%d.%d.%s.%s"%(tt, ll, var, xy)
                    name2 = get_tensor_name(name, {"final_layer":(ll==nlayer_fit-1)})
                    nameList.append(name)
                    nameList2.append(name2)
    return nameList, nameList2

def get_tensor_list(nameList):
    tensorlist = []
    for name in nameList:
        tensor = tf.get_default_graph().get_tensor_by_name(name)
        tensorlist.append(tensor)
    return tensorlist

def get_tensor_value(sess, feed_dict, tensorlist, nameList=[]):
    val = sess.run(tensorlist, feed_dict=feed_dict)

    if len(nameList) < len(tensorlist):
        return val 
    else:
        res = {}
        for ii in range(len(nameList)):
            res[nameList[ii]] = val[ii]
        return res
    

def main_simple(argvs):
    """:仅仅预测基本的输入输出
    """
    # 读入参数
    tt = 0
    json_fn = argvs[tt]; tt += 1
    atom_fn = argvs[tt]; tt += 1
    is_gen  = argvs[tt]; tt += 1

    debug, sess, atoms, feed_dict, cf = init(json_fn, atom_fn)

    #
    cf = read_cf()
    ntype = cf['ox123129134125114112118123114']
    sel = cf['ox128114121117103'][0:ntype]
    print("INFO: sel \n", sel)

    # get namelist
    nameList = [] # 便于标记使用的别名
    nameList2 = [] # tensor的名字
    ss = ''
    ss = 't_coord,t_type,t_natoms_vec,t_box,t_default_mesh,'
    ss += 'o_energy,o_force,o_virial,o_atom_energy,o_atom_virial,'
    ss += 'o_rmat,o_rmat_deriv,o_rij,o_nlist,'
    ss += 'o_descriptor,'
    ss = ss.rstrip(',')
    ss = ss.split(',')
    nameList.extend(ss)
    ss = [si+':0' for si in ss]
    nameList2.extend(ss)

    # name to tensor
    print("INFO: Name2Tensor")
    tensorlist = get_tensor_list(nameList2)

    ## predict value
    print("INFO: Sess Run")
    res = get_tensor_value(sess, feed_dict, tensorlist, nameList)

    np.save('res.npy', [res], allow_pickle=True)

    # code input of vcs
    if is_gen == '1':
        main_code_input()
    


def main_code_input():
    """:code input of vcs
    """
    cf = read_cf()
    NSEL = cf['ox091096082089']
    Na = cf['ox091110']
    NI = cf['ox091086']
    ntype = cf['ox123129134125114112118123114']
    NI2 = int(np.sum(cf['ox128114121117103'][0:ntype]))

    model = np.load('fpga_model.npy', allow_pickle=True)[0]
    cfg = model['cfg']
    fea = model['fea']
    gra = model['gra']
    wfp = model['wfp']
    wbp = model['wbp']

    # output cfg and fea and gra
    head = code_vcs.code_head(len(cfg), 0, len(fea), len(gra), 0, 0)
    scfg = [head]
    scfg.extend(cfg)
    scfg.extend(fea)
    scfg.extend(gra)
    np.savetxt('lmp1_hex.txt', scfg, fmt="%s")

    # output wfp and wbp
    n = len(wfp) // NSEL
    wfp = [wfp[ii][-72//4:] for ii in range(len(wfp))]
    wbp = [wbp[ii][-72//4:] for ii in range(len(wbp))]
    swfp = [ ''.join((wfp[n*ii:n*(ii+1)])[::-1]) for ii in range(NSEL)]
    swbp = [ ''.join((wbp[n*ii:n*(ii+1)])[::-1]) for ii in range(NSEL)]

    np.savetxt('wfp_hex.txt', swfp, fmt="%s")
    np.savetxt('wbp_hex.txt', swbp, fmt="%s")

    # output lst and atm
    res = np.load('res.npy', allow_pickle=True)[0]

    cfl = {'Na':Na, 'NI':NI, 'NI2':NI2}
    idx = resort_lst(res['o_nlist'], cfl)
    res['o_nlist'] = sort_by_lst(res['o_nlist'], idx, cfl, -1)

    code_vcs.save_vcs_in(res)


def main_test_energy_force(argvs):
    """当原子位移时，得到的能量和受力，确认是否为求导关系
    """
    # 读入参数
    tt = 0
    json_fn = argvs[tt]; tt += 1
    atom_fn = argvs[tt]; tt += 1
    idx = int(argvs[tt]); tt += 1 #测试的原子
    rng = int(argvs[tt]); tt += 1 #测试点数 [-rng, rng]
    ksx = float(argvs[tt]); tt += 1 #平移倍数

    debug, sess, atoms, feed_dict, cf = init(json_fn, atom_fn)

    #
    cf = read_cf()
    ntype = cf['ox123129134125114112118123114']
    sel = cf['ox128114121117103'][0:ntype]
    print("INFO: sel \n", sel)

    # get namelist
    nameList = [] # 便于标记使用的别名
    nameList2 = [] # tensor的名字
    ss = ''
    ss += 'o_energy,o_force,'
    ss = ss.rstrip(',')
    ss = ss.split(',')
    nameList.extend(ss)
    ss = [si+':0' for si in ss]
    nameList2.extend(ss)

    # name to tensor
    print("INFO: Name2Tensor")
    tensorlist = get_tensor_list(nameList2)

    ## predict value
    print("INFO: Sess Run")

    shift = np.arange(-rng, rng+1) * ksx
    vx = atoms[idx].x

    # res.npy
    res = {
        'arg':{
        'idx':idx,
        'rng':rng,
        'ksx':ksx
        }}
    for ii in range(rng*2+1):
        print("#INFO: %d | %d "%(ii,rng*2+1))
        
        atoms[idx].x = vx + shift[ii]
        _, feed_dict = init_atom(atom_fn, atoms)
        resi = get_tensor_value(sess, feed_dict, tensorlist, nameList)
        res["%f"%shift[ii]] = resi

    print("#INFO: save result in res.npy")
    np.save('res.npy', [res], allow_pickle=True)

    # res.txt
    x, e, f = [], [], []
    for key in res:
        if key == 'arg': continue
        x.append(float(key))
        e.append(res[key]['o_energy'][0])
        f.append(np.reshape(res[key]['o_force'], [-1, 3])[idx][0])
    dat = [[x[ii], e[ii], f[ii]] for ii in range(len(x))]

    print("#INFO: save result in res.txt with colunmn: x(Ang), e(eV), f(eV/Ang)")
    np.savetxt('res.txt', dat)

def main(argvs):
    """:测试网络中的每一个参数，用于vcs debug
    """
    # 读入参数
    tt = 0
    json_fn = argvs[tt]; tt += 1
    atom_fn = argvs[tt]; tt += 1

    debug, sess, atoms, feed_dict, cf = init(json_fn, atom_fn)
    natoms_vec = feed_dict[debug.place_holders['natoms_vec']]

    #
    cf = read_cf()
    NI = cf['ox091086']
    ntype = cf['ox123129134125114112118123114']
    ntypex = cf['ox123129134125114133']
    same_net = cf['ox128110122114108123114129']
    nlayer_fit = cf['ox123121110134114127108115118129']
    sel = cf['ox128114121117103'][0:ntype]
    natom = natoms_vec[0]
    print("INFO: sel \n", sel)

    # get namelist
    nameList, nameList2 = get_tensor_namelist()

    # name to tensor
    print("INFO: Name2Tensor")
    tensorlist = get_tensor_list(nameList2)

    ## predict value
    print("INFO: Sess Run")
    res = get_tensor_value(sess, feed_dict, tensorlist, nameList)
    
    ## 数据拼接 S, SR, G
    print("INFO: Merge S, SR, G")
    nameList3 = []
    for var in 's,sr,g'.split(','):
        for xy in 'x,y,dy,dydx,dx'.split(','):
            vss = []
            for tt in range(ntypex):
                vs = []
                for tt2 in range(ntype):
                    name = "fea.%d.%d.%s.%s"%(tt, tt2, var, xy)
                    v = res[name]
                    n = natoms_vec[tt+1] if same_net else natoms_vec[tt+2]
                    s = sel[tt2]
                    v = np.reshape(v, [n, s, -1])
                    vs.append(v)
                vs = np.hstack(vs)
                vss.append(vs)
            
            name = "fea.%s.%s"%(var, xy)
            vss = np.vstack(vss)
            res[name] = vss
            nameList3.append(name)

    ## 数据拼接 GR, DGR, D, dD, R4
    print("INFO: Merge GR, R4")
    for var in "gr,dgr,d,dd,r4.y,r4.dx".split(','):
        vs = []
        for tt in range(ntypex):
            name = "fea.%d.%s"%(tt, var)
            v = res[name]
            n = natoms_vec[tt+1] if same_net else natoms_vec[tt+2]
            v = np.reshape(v, [n, -1])
            vs.append(v)
        name = "fea.%s"%(var)
        vs = np.vstack(vs)
        res[name] = vs
    nameList3.append('fea.r4.y')
    nameList3.append('fea.r4.dx')

    ## 数据拼接 FIT
    print("INFO: Merge FIT")
    for ll in range(nlayer_fit):
        for var in 'wx,tanh,y'.split(','):
            for xy in 'y,dx'.split(','):
                vs = []
                for tt in range(ntype):
                    name = "fit.%d.%d.%s.%s"%(tt, ll, var, xy)
                    v = res[name]
                    n = natoms_vec[tt+2]
                    v = np.reshape(v, [n, -1])
                    vs.append(v)
                name = "fit.%d.%s.%s"%(ll, var, xy)
                vs = np.vstack(vs)
                res[name] = vs


    ## lst 重排
    print("INFO: Resort LST")
    cfl = {'Na':natom, 'NI':NI, 'NI2':int(np.sum(sel))}
    idx = resort_lst(res['o_nlist'], cfl)

    nameList3.extend('o_rmat,o_rmat_deriv,o_rij,o_nlist'.split(','))
    for key in nameList3:
        res[key] = sort_by_lst(res[key], idx, cfl, -1 if key == 'o_nlist' else 0)
    
    ## save the data to file
    print("INFO: SAVE DATA")
    code_vcs.save_descrpt(res)
    code_vcs.save_cal_fij(res)
    code_vcs.save_feaNet(res)
    code_vcs.save_fea(res)
    code_vcs.save_fit(res)
    code_vcs.save_out(res)


def help():
    print('predict: train.json atoms.xsf [is_gen_vcs_in(1|0)]')
    print('debug: train.json atoms.xsf')
    print('testef: train.json atoms.xsf idx rng ksx')

if __name__ == "__main__":
    argvs = sys.argv[1:]
    mod = argvs[0] if len(argvs) > 0 else '--help'
    if mod == 'predict':
        main_simple(argvs[1:])
    if mod == 'debug':
        main(argvs[1:])
    if mod == 'testef':
        main_test_energy_force(argvs[1:])
    if mod == '--help':
        help()
        
        



