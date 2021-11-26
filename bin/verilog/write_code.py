
import struct
from env2 import np
from ase.io import write, read
import scipy.io as sio

import json
from ase import Atom
from ase import Atoms
from ase import units
import sys
import os

from WCode import WCode
from opt import cal_map_param, get_GRRG_idx, opt_too_small_w
from opt import code_st, split_factor

from code_hb import val2bin, bin2hex
from network import get_config, set_config

from env2 import intt

"""
//
// ====================================================================

# 功能
读取配置文件参数，并自动生成verilog代码参数

// ====================================================================
//
"""

def set_cf_0(fn):
    print("INFO: read config file\n", fn)
    if fn.endswith(".json"):
        fr = open(fn, 'r')
        jdata = json.load(fr)
        fr.close()
        cf = jdata
    else:
        cf = np.load(fn, allow_pickle=True)[0]
    return cf

def set_cf_1(cf):
    """
    # 功能
    添加网络参数
    """
    fn = cf['weight_file']
    print("INFO: read network file\n", fn)
    # 读取网络，获取网络参数
    net = np.load(fn, allow_pickle=True)[0]
    cf.update(net)

    # 处理一般参数
    cf['ox091089078102082095'] = cf['ox123121110134114127108115118129']
    cf['NTYPE'] = cf['ox123129134125114112118123114']
    cf['NTYPEX'] = cf['ox123129134125114133']
    cf['SAME_NET'] = 1 if cf['ox128110122114108123114129'] else 0
    cf['ox128114121117103'].extend([0,0,0,0])
    for ii in range(4): cf['SEL%d'%ii] = cf['ox128114121117103'][ii]
    cf['element'] = [Atom(ele) for ele in cf['ox128134122111124121']]
    cf['type'] = [ele.number for ele in cf['element']]

    rc = cf['ox127112']
    cf['RC'] = qq(rc, cf['ox091079086097108081078097078108083089'], True)
    rc2 = cf['ox127112'] ** 2
    cf['RC2'] = qq(rc2, cf['ox091079086097108081078097078108083089'], True)

    cf['NI_S'] = int(np.ceil(cf['ox091086'] / cf['ox091096097081090080085086070']))
    cf['M1_S'] = int(np.ceil(cf['ox090062'] / cf['ox091096097081090080085086070']))
    cf['NIS'] = int(cf['NI_S'] * cf['ox091096097081090080085086070'])
    cf['M1S'] = int(cf['M1_S'] * cf['ox091096097081090080085086070'])

    # 计算R2U的参数
    xps, ks, bs, xps2, ks2, bs2 = cal_map_param(cf['ox127112'])
    cf['map_param'] = [xps, ks, bs, xps2, ks2, bs2]
    cf['R2U_XS'] = qq(xps[:,0], cf['ox091079086097108081078097078108083089'], True)
    cf['R2U_KS'] = -np.round(np.log(ks) / np.log(2))
    cf['R2U_BS'] = qq(bs, cf['ox091079086097108081078097078108083089'], True)

    # 生成fitNet参数模块的参数
    cf['ox091096082089'] = cf['ox091097102093082108090078101080086091082071097079067'] * cf['ox091096097081090080085086070']
    cf['ox091091092081082108083086097096'].insert(0, cf['ox090062'] * cf['ox090063'])
    cf['ox091091092081082108083086097096'].extend([0, 0, 0, 0])
    cf['ox091091092081082108083086097096'][cf['ox091089078102082095']] = 1

    ## 编码fitNet的权值(二进制)
    b_b = []
    s_b = []
    w_b = []
    for ll in range(4):
        b_b_t = []
        s_b_t = []
        w_b_t = []
        for tt in range(cf['ox091097102093082108090078101080086091082071097079067']):
            if tt >= cf['NTYPE']:
                cf['fit_t%d_l%d_b'%(tt, ll)] = cf['fit_t%d_l%d_b'%(0, ll)] * 0
                cf['fit_t%d_l%d_m'%(tt, ll)] = cf['fit_t%d_l%d_m'%(0, ll)] * 0
            # value
            vb = cf['fit_t%d_l%d_b'%(tt, ll)]
            vw = cf['fit_t%d_l%d_m'%(tt, ll)]
            # check
            vb = check_range(vb, cf['ox128114129108127123116108115118129108111117103118097116112105097104107118097'], 'fit_t%d_l%d_b'%(tt, ll))
            # code
            bb = gen_code_b(vb, 8+cf['ox091079086097108081078097078108083089'], cf['ox091079086097108081078097078108083089'], tt, ll)
            sb, wb = gen_code_w(vw, cf['ox091098090108100089091063'], cf['ox091079086097108081078097078108083089'], cf['ox091079086097108100089091063080068075086097089078080'], tt, ll)
            b_b_t.append(bb)
            s_b_t.append(sb)
            w_b_t.append(wb)
        b_b.append(b_b_t)
        s_b.append(s_b_t)
        w_b.append(w_b_t)
    
    bfps = []
    bbps = []
    for ss in range(cf['ox091096082089']):
        tt = ss // cf['ox091096097081090080085086070']
        sc = ss % cf['ox091096097081090080085086070']
        sr = ss % cf['ox091096097081090080085086070']
        bfp = ''
        bbp = ''
        for ll in range(4):
            nr = cf['ox091091092081082108083086097096'][ll]
            nc = cf['ox091091092081082108083086097096'][ll+1]
            nrs = int(np.ceil(nr/cf['ox091096097081090080085086070']))
            ncs = int(np.ceil(nc/cf['ox091096097081090080085086070']))
            nw = cf['ox091098090108100089091063']
            nx = cf['ox091096097081090080085086070'] if ll == 3 else 1
            xs = 0 if ll == 3 else 1
            #* fp *#
            bi = [s_b[ll][tt][rr][sc*ncs*xs+cc][ww] for cc in range(ncs) for rr in range(nr) for ww in range(nw)]; bi.reverse(); bfp = ''.join(bi) + bfp
            bi = [w_b[ll][tt][rr][sc*ncs*xs+cc][ww] for cc in range(ncs) for rr in range(nr) for ww in range(nw)]; bi.reverse(); bfp = ''.join(bi) + bfp
            bi = [b_b[ll][tt][sc*ncs*xs+cc] for cc in range(ncs)]; bi.reverse(); bfp = ''.join(bi) + bfp
            #* bp *#
            bi = [s_b[ll][tt][sr*nrs+rr][cc][ww] for rr in range(nrs) for cc in range(nc) for ww in range(nw)]; bi.reverse(); bbp = ''.join(bi) * nx + bbp
            bi = [w_b[ll][tt][sr*nrs+rr][cc][ww] for rr in range(nrs) for cc in range(nc) for ww in range(nw)]; bi.reverse(); bbp = ''.join(bi) * nx + bbp
            bi = [b_b[ll][tt][sc*ncs*xs+cc] for cc in range(ncs)]; bi.reverse(); bbp = ''.join(bi) + bbp

        bfps.append(bfp)
        bbps.append(bbp)
    cf['bfps'] = bfps
    cf['bbps'] = bbps
    return cf

def set_cf_2(cf):
    """
    # 功能
    添加针对仿真系统的参数
    """
    pass
    return cf     

def set_cf_3(cf):
    """
    # 功能
    添加位宽参数
    """
    print("INFO: generate more code configure\n")
    cf['ox091079086097108080095081'] = cf['ox091079086097108081078097078080068075086097070067086'] * 3
    cf['ox091079086097108089096097'] = intt(np.ceil(np.log(cf['ox091110101080099'])/np.log(2)))
    cf['ox091079086097108078097092090080068075086097067086081'] = cf['ox091079086097108096093082'] + cf['ox091079086097108080095081']
    cf['ox091079086097108089092091084108078097092090'] = cf['ox091079086097108096093082'] + cf['ox091079086097108089092091084108081078097078'] * 3 + cf['ox091079086097108081086083083108081078097078'] * 3
    cf['ox091079086097108095086087'] = cf['ox091079086097108081078097078108083089'] + 5
    cf['ox091079086097108096098090'] = cf['ox091079086097108081078097078108083089'] + 8
    cf['NBIT_DIV_NI'] = intt(np.round(np.log(cf['ox091086101080075'])/np.log(2)))
    cf['NBIT_DATA_NI'] = cf['ox091079086097108081078097078080068075086097070067086'] + cf['NBIT_DIV_NI']
    
    return cf

def set_cf_4(cf):
    """
    # 功能
    读入featureNet中的映射
    """
    fn = cf['map_file']
    print("INFO: read map file\n", fn)
    # 读取映射文件
    maps = np.load(fn, allow_pickle=True)[0]
    cf.update(maps)

    # 一般参数
    nbit = intt(cf['ox091079086097108083082078'] - cf['ox091079086097108083082078108083089080068075086097072071067097072'] - 1)

    # shift_gs中的参数定义
    M1 = cf['ox090062']
    ntypex = cf['ox123129134125114133']
    ntype = cf['ox123129134125114112118123114']
    arr_gs = np.zeros([4, 4, M1])
    for tt in range(ntypex):
        for tt2 in range(ntype):
            postfix = '_t%d_t%d'%(tt,tt2)
            s = maps['table_s'+postfix]
            G = maps['table_G'+postfix]
            arr_gs[tt,tt2,:] = s[0]*G[0,:]
    
    g_b = gen_code_gs(arr_gs, 2*cf['ox091079086097108083082078108083089080068075086097072071067097072'])
    g_b = ''.join(g_b[::-1])
    cf['g_b'] = g_b
    cf['sfea'] = maps['sfea']
    cf['sgra'] = maps['sgra']
    return cf

def set_cf_5(cf, arvgs):
    print("INFO: change parameter")
    pars = arvgs.split(',')
    for ii in range(len(pars)):
        p = pars[ii].split(':')
        name = p[0]
        value = p[1]
        if value.isdigit():
            cf[name] = int(value)
            t = int
        else:
            t = type(cf[name])
            cf[name] = t(value)
        
        print(name, ':', value)
    return cf

def write_cf(cf, merge=False, dic_fn=''):
    """:将配置输出成16进制文件
    """
    ntype = cf['NTYPE']
    ntype2 = cf['ox091097102093082108090078101080086091082071097079067']
    ln2_NIX = int(np.ceil(np.log2(cf['ox091086101080075'])))

    # cfg
    bs = ''
    bs = val2bin(cf['R2U_XS'][0], cf['ox091079086097108081078097078080068075086097070067086']) + bs
    bs = val2bin(cf['R2U_XS'][1], cf['ox091079086097108081078097078080068075086097070067086']) + bs
    bs = val2bin(cf['RC2']      , cf['ox091079086097108081078097078080068075086097070067086']) + bs
    bs = val2bin(cf['R2U_KS'][0], cf['ox091079086097108081078097078080068075086097070067086']) + bs
    bs = val2bin(cf['R2U_KS'][1], cf['ox091079086097108081078097078080068075086097070067086']) + bs
    bs = val2bin(cf['R2U_KS'][2], cf['ox091079086097108081078097078080068075086097070067086']) + bs
    bs = val2bin(cf['R2U_BS'][0], cf['ox091079086097108081078097078080068075086097070067086']) + bs
    bs = val2bin(cf['R2U_BS'][1], cf['ox091079086097108081078097078080068075086097070067086']) + bs
    bs = val2bin(cf['R2U_BS'][2], cf['ox091079086097108081078097078080068075086097070067086']) + bs
    bs = val2bin(cf['ox128114121117103'][0], cf['ox091079086097108089096097']) + bs
    bs = val2bin(cf['ox128114121117103'][1], cf['ox091079086097108089096097']) + bs
    bs = val2bin(cf['ox128114121117103'][2], cf['ox091079086097108089096097']) + bs
    bs = val2bin(cf['ox128114121117103'][3], cf['ox091079086097108089096097']) + bs
    bs = cf['g_b'] + bs

    bn = val2bin(ln2_NIX, cf['ox091079086097108096085086083097'])
    bs = bn + bs

    s1 = save_bin2hex(bs, 64, None if merge else "out/cfg_hex.txt")
    s2 = save_bin2hex(cf['bfps'], 72, None if merge else "out/wfp_hex.txt")
    s3 = save_bin2hex(cf['bbps'], 72, None if merge else "out/wbp_hex.txt")

    # write data to file
    if merge:
        # extend the length for increasing ntype to 4
        # s2 = ext_len(s2, ntype, ntype2)
        # s3 = ext_len(s3, ntype, ntype2)
        s4 = cf['sfea']
        s4 = ext_len(s4, ntype, ntype2)
        s5 = cf['sgra']
        s5 = ext_len(s5, ntype, ntype2)

        # extend length
        s1 = ext_hex(s1)
        s2 = ext_hex(s2)
        s3 = ext_hex(s3)
        s4 = ext_hex(s4)
        s5 = ext_hex(s5)

        s = []
        s.extend(s1)
        s.extend(s2)
        s.extend(s3)
        s.extend(s4)
        s.extend(s5)

        # write the string (hex) to binary file
        with open(cf['fpga_model_file'], 'wb') as fp:
            for si in s:
                for ii in range(len(si)//2):
                    v = int(si[2*ii:2*(ii+1)],16)
                    v = struct.pack('B', v)
                    fp.write(v)
        
        # save the dic for debug
        if dic_fn != '':
            dic = {}
            dic['cfg'] = s1
            dic['wfp'] = s2
            dic['wbp'] = s3
            dic['fea'] = s4
            dic['gra'] = s5
            np.save(dic_fn, [dic], allow_pickle=True)

def ext_len(s, nt, nt2):
    dnt = nt2 - nt
    nl = len(s)
    nl2 = len(s[0])
    ds = '0' * nl2
    dn = (nl // nt) * dnt
    s.extend([ds for ii in range(dn)])
    return s

def save_bin2hex(bs, nbit, fn=None):
    if type(bs) != list:
        bs = [bs]
    hsls = []
    for ii in range(len(bs)):
        bsi = bs[ii]
        hs = bin2hex(bsi) 
        hsl = []
        nhex = int(np.ceil(nbit/4))
        n = int( np.ceil(len(hs)/(nhex)) )
        if (ii==0): print("INFO: save_bin2hex.n", n)
        hs = ('0' * (n*nhex - len(hs))) + hs 
        for ii in range(n):
            hsl.append(hs[ii*nhex:(ii+1)*nhex])
        hsl = hsl[::-1]
        hsls.extend(hsl)
    if fn != None:
        print("INFO: save config binary file\n", fn)
        np.savetxt(fn, hsls, fmt="%s")
        return ''
    else:
        return hsls

def write_code(cf, tmp_path, out_path, fns):
    # decode
    lib_path = os.path.dirname(os.path.realpath(__file__))
    decode_dic = np.load(lib_path+'/../data/decode.npy', allow_pickle=True)[0]
    for key in decode_dic.keys():
        cf[key] = cf[decode_dic[key]]
    
    # write code
    wcObj = WCode(tmp_path, out_path)
    for m in fns:
        m = m.replace('.v','')
        wcObj.instantiate(m, cf, name=m)

# 子函数
# ====================================================================

def qq(x, nbit=14, is_round=False):
    if type(nbit) == str:
        print(nbit)
    prec = 2**nbit
    x = np.array(x)
    if is_round:
        return intt(np.round(x * prec))
    else:
        return intt(np.floor(x * prec))

def v2s2v(v):
    v2 = []
    s = ""
    for vi in v:
        si = "%16.10f"%vi
        s += si + " "
        v2.append(float(si))
    print(s)
    return v2

def pow_2n(xv):
    v = np.log2(1.5)
    eps = 1e-12
    xlog = np.log2(np.abs(xv) + eps)
    n = np.ceil(xlog - v)
    v2n = np.power(2.0, n)
    return v2n, n

def gen_code_w(w, NUM_WLN2, NBIT_DATA_FL, NBIT_WLN2, tt, ll):
    """:将权值分解编码
    """
    th = 2**(-NBIT_DATA_FL)
    sh = w.shape
    nr = sh[0]
    nc = sh[1]
    signs = np.zeros([nr * nc, NUM_WLN2])
    w2n   = np.zeros([nr * nc, NUM_WLN2])

    # w => s1*2^n1 + s2*2^n2 + ... sn*2^nn
    # s > 0编码为1
    # s < 0编码为-1
    # |vw| < prec, s编码为0
    vw = w.reshape([-1]).copy()
    for iw in range(NUM_WLN2):
        v2n, w2n[:,iw] = pow_2n(vw)
        signs[:,iw] = -1
        s = vw > 0
        signs[s,iw] = 1
        s = v2n < th
        signs[s,iw] = 0
        w2n[s,iw] = -NBIT_DATA_FL
        vw -= signs[:,iw] * v2n
    
    # 转换编码
    arrs = np.zeros([nr * nc, NUM_WLN2])
    arrw = np.zeros([nr * nc, NUM_WLN2])
    for ii in range(nr * nc):
        for jj in range(NUM_WLN2):
            arrs[ii,jj] = 3 if signs[ii, jj] < 0 else signs[ii, jj]
            arrw[ii,jj] = NBIT_DATA_FL + w2n[ii, jj]
    
    # 编码为二进制
    arrs = np.reshape(arrs, [nr, nc, NUM_WLN2])
    arrw = np.reshape(arrw, [nr, nc, NUM_WLN2])

    Ss = []
    Ws = []
    for rr in range(nr):
        Ss1 = []
        Ws1 = []
        for cc in range(nc):
            Ss2 = []
            Ws2 = []
            for ww in range(NUM_WLN2):
                s = val2bin(arrs[rr, cc, ww], 2)
                w = val2bin(arrw[rr, cc, ww], NBIT_WLN2)
                Ss2.append(s)
                Ws2.append(w)
            Ss1.append(Ss2)
            Ws1.append(Ws2)
        Ss.append(Ss1)
        Ws.append(Ws1)
    return Ss, Ws

def gen_code_b(vb, NBIT_SUM, NBIT_DATA_FL, tt, ll):
    prec = 2**NBIT_DATA_FL
    sh = vb.shape
    nr = sh[0]
    arrb = np.round(vb*prec)
    arrb += 2**NBIT_SUM

    # 编码为二进制
    Bs = []
    for ic in range(nr):
        b = val2bin(arrb[ic], NBIT_SUM)
        Bs.append(b)
    return Bs

def gen_code_gs(vgs, NBIT_DATA_FL):
    sh = vgs.shape
    M1 = sh[2]
    prec = 2 ** NBIT_DATA_FL
    arr_gs = np.round(vgs * prec)
    arr_gs += 2**27

    GSs = []
    for tt in range(1):
        for tt2 in range(4):
            for ii in range(M1):
                gs = val2bin(arr_gs[tt, tt2, ii], 27)
                GSs.append(gs)
    return GSs

def check_range(v, rng, name=''):
    print("#INFO: CHECK_RANGE: %s"%(name))
    check = np.abs(v) > rng 
    if np.sum(check) > 0:
        print("#INFO: many value of %s are bigger than range %d"%(name, rng))
        print("#INFO: clip the value to range %d"%(rng))
        print(v)
        v = np.maximum(v, -rng)
        v = np.minimum(v,  rng)
    return v

def ext(fn):
    fr = open(fn,'r')
    lines = fr.readlines()
    fr.close()

    lines = [line.replace('\n','') for line in lines]
    n = len(lines[0])
    s = '0' * (128-n)
    lines = [s + line for line in lines]

    # print(lines)
    fn2 = fn.split('.')
    fn2[-2] += '_ext'
    fn2 = '.'.join(fn2)

    fw = open(fn2,'w')
    for line in lines:
        fw.writelines(line+'\n')
    fw.close()

def ext_hex(hs):
    s = '0' * (128 - len(hs[0]))
    hs = [s + si for si in hs]
    return hs

def ext_atom(type_fn, type_map_fn, set_path, idx):
    types = np.int64(np.loadtxt(type_fn))
    type_map = [line.replace('\n', '') for line in open(type_map_fn, 'r').readlines()]
    print("#INFO: The type of atoms is", types)
    print("#INFO: The type_map of atoms is", type_map)
    coords = np.load(set_path+'/coord.npy')
    boxs = np.load(set_path+'/box.npy')
    nframe = len(boxs)
    if idx >= nframe:
        print("#INFO: The number of frame of atoms is %s (<= idx %s)"%(nframe, idx))
    else:
        coord = np.reshape(coords[idx], [-1, 3])
        box = np.reshape(boxs[idx], [3, 3])
        syml = ''.join([type_map[ii] for ii in types])
        atoms = Atoms(syml, positions=coord, cell=box, pbc=[1,1,1])
        write('atoms_%d.xsf'%idx, atoms)

# 主函数
# ====================================================================

# -- mian --
def main(argvs):
    if (argvs[0]) == 'freeze':
        config_file = get_config('verilog_file')
        weight_file = argvs[1]
        config_npy_file = argvs[2]
        cf = set_cf_0(config_file)
        cf['weight_file'] = weight_file
        cf = set_cf_1(cf)
        cf = set_cf_3(cf)
        np.save(config_npy_file, [cf])
    
    if (argvs[0]) == 'wrap':
        config_file = argvs[1]
        map_file = argvs[2]
        fpga_model_file = argvs[3]
        debug_dic_file = argvs[4] if len(argvs) >= 5 else ''
        cf = set_cf_0(config_file)
        cf['map_file'] = map_file
        cf['fpga_model_file'] = fpga_model_file
        cf = set_cf_4(cf)
        write_cf(cf, True, debug_dic_file)
    
    if (argvs[0] == 'code'):
        config_file = argvs[1]
        verilog_tmp_path = argvs[2]
        verilog_out_path = argvs[3]
        verilog_files = argvs[4:]
        cf = set_cf_0(config_file)
        write_code(cf, verilog_tmp_path, verilog_out_path, verilog_files)
    
    if (argvs[0] == 'ext_atm'):
        type_fn = argvs[1]
        type_map_fn = argvs[2]
        set_path = argvs[3]
        idx = int(argvs[4])
        ext_atom(type_fn, type_map_fn, set_path, idx)

        
