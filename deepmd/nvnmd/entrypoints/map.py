
import numpy as np
import matplotlib.pyplot as plt

from deepmd.env import tf 
from deepmd.nvnmd.utils.fio import Fio, FioDic
from deepmd.nvnmd.utils.config import nvnmd_cfg
from deepmd.nvnmd.utils.weight import get_r2u_u2r_param, get_normalize, get_filter_weight 
from deepmd.nvnmd.utils.network import get_sess 

from deepmd.nvnmd.data.data import jdata_deepmd_input

class Map:

    def __init__(self, 
        config_file: str, 
        weight_file: str, 
        map_file: str
        ) -> None:
        self.config_file = config_file
        self.weight_file = weight_file
        self.map_file = map_file

        jdata = jdata_deepmd_input['nvnmd']
        jdata['config_file'] = config_file
        jdata['weight_file'] = weight_file
        jdata['enable'] = True

        nvnmd_cfg.init_from_jdata(jdata)
        map_table = self.build_map()

    def plot_map(self, dic, path='nvnmd/map'):
        x = dic['r2']
        keys = 's,sr,G'.split(',')
        keys2 = 'ds_dr2,dsr_dr2,dG_dr2'.split(',')
        for key in (keys+keys2):
            for key2 in dic.keys():
                if key2.startswith(key):
                    v = dic[key2]

                    file_name = f'{path}/{key2}.png'
                    Fio().create_file_path(file_name)
                    plt.plot(x, v)
                    plt.xlabel('R^2')
                    plt.ylabel('y')
                    plt.savefig(file_name)
                    plt.close()

    def build_map(self):
        ntypex = nvnmd_cfg.dscp['ntypex']
        ntype = nvnmd_cfg.dscp['ntype']
        NBIT_FEA_FL = nvnmd_cfg.nbit['NBIT_FEA_FL']

        dic = self.run_u2s()
        dic.update(self.run_s2G(dic))

        dic['dG_dr2'] = []
        for tt in range(ntypex):
            for tt2 in range(ntype):
                dic['dG_dr2'].append(dic['dG_ds'][tt*ntype+tt2] * dic['ds_dr2'][tt])
        
        prec = 2**NBIT_FEA_FL
        for tt in range(ntypex):
            dic['s'][tt][0] = np.round(dic['s'][tt][0] * prec) / prec
            for tt2 in range(ntype):
                dic['G'][tt*ntype+tt2][0] = np.round(dic['G'][tt*ntype+tt2][0] * prec) / prec

        def qqq(dat, nbit, is_round=False):
            dat = dat if type(dat) == list else [dat]
            prec = 2 ** (nbit-1)
            #
            if is_round: dat = [np.round(dati * prec) / prec for dati in dat]
            else: dat = [np.floor(dati * prec) / prec for dati in dat]
            return dat

        keys = 's,sr,G,ds_dr2,dsr_dr2,dG_dr2'.split(',')
        maps = {}
        for key in keys:
            val = qqq(dic[key], NBIT_FEA_FL)
            maps[key] = val
        
        n = len(dic['u'])
        maps2 = {}
        maps2['u'] = dic['u']
        maps2['r2'] = dic['r2']
        for tt in range(ntypex):
            for tt2 in range(ntype):
                postfix = f'_t{tt}_t{tt2}'
                for key in keys:
                    if 'G' not in key:
                        maps2[key+postfix] = maps[key][tt].reshape([n, -1])
                    else:
                        maps2[key+postfix] = maps[key][tt*ntype+tt2].reshape([n, -1])
        self.map = maps2
        # self.plot_map(self.map)

        FioDic().save(self.map_file, self.map)
        return self.map 

    def build_seg(self, x, xps, ks, bs):
        x1 = tf.clip_by_value(x, xps[0], xps[1])
        x2 = tf.clip_by_value(x, xps[1], xps[2])
        x3 = tf.clip_by_value(x, xps[2], xps[3])

        y = (x1-xps[0])*ks[0] + (x2-xps[1])*ks[1] + (x3-xps[2])*ks[2]
        return y

    def build_u2r(self, u):
        rcut = nvnmd_cfg.dscp['rcut']

        xps, ks, bs, xps2, ks2, bs2 = get_r2u_u2r_param(rcut)
        xps2 = xps2[:,0].tolist()
        xps2.insert(0,0)
        xps2.append(1.0)

        r2 = self.build_seg(u, xps2, ks2, bs2) * (rcut**2)
        return r2
    
    def build_r2s(self, r2):
        limit = nvnmd_cfg.dscp['rc_lim']
        rmin = nvnmd_cfg.dscp['rcut_smth']
        rmax = nvnmd_cfg.dscp['rcut']
        ntype = nvnmd_cfg.dscp['ntype']
        avg, std = get_normalize(nvnmd_cfg.weight)

        r = tf.sqrt(r2)
        r_ = tf.clip_by_value(r, rmin, rmax)
        r__ = tf.clip_by_value(r, limit, rmax) # 小于此limit的值保持恒定
        uu = (r_ - rmin) / (rmax - rmin)
        vv = uu*uu*uu * (-6 * uu*uu + 15 * uu - 10) + 1

        sl = []
        srl = []

        for tt in range(ntype):
            s = vv / r__
            sr = s / r__
            s = tf.reshape(s, [-1, 1])
            sr = tf.reshape(sr, [-1, 1])
            s = (s - avg[tt,0]) / std[tt,0]
            sr = sr / std[tt,1]
            sl.append(s)
            srl.append(sr)
        return sl, srl
    
    def build_ds_dr(self, r2, s, sr):
        ntype = nvnmd_cfg.dscp['ntype']

        ds_drl = []
        dsr_drl = []
        for tt in range(ntype):
            si = s[tt]
            sri = sr[tt]
            ds_dr = tf.gradients(si, r2)
            dsr_dr = tf.gradients(sri, r2)
            ds_drl.append(ds_dr[0])
            dsr_drl.append(dsr_dr[0])
        return ds_drl, dsr_drl

    def build_s2G(self, s):
        ntypex = nvnmd_cfg.dscp['ntypex']
        ntype = nvnmd_cfg.dscp['ntype']

        activation_fn = tf.tanh
        outputs_size = nvnmd_cfg.dscp['NNODE_FEAS']

        xyz_scatters = []
        for tt in range(ntypex):
            for tt2 in range(ntype):
                xyz_scatter = s[tt]
                for ll in range(1, len(outputs_size)):
                    w, b = get_filter_weight(nvnmd_cfg.weight, tt, tt2, ll)
                    if outputs_size[ll] == outputs_size[ll-1]:
                        xyz_scatter += activation_fn(tf.matmul(xyz_scatter, w) + b)
                    elif outputs_size[ll] == outputs_size[ll-1] * 2: 
                        xyz_scatter = tf.concat([xyz_scatter,xyz_scatter], 1) + activation_fn(tf.matmul(xyz_scatter, w) + b)
                    else:
                        xyz_scatter = activation_fn(tf.matmul(xyz_scatter, w) + b)
                xyz_scatters.append(xyz_scatter)
        return xyz_scatters

    def build_dG_ds(self, G, s):
        ntypex = nvnmd_cfg.dscp['ntypex']
        ntype = nvnmd_cfg.dscp['ntype']
        M1 = nvnmd_cfg.dscp['M1']

        dG_ds = []
        for tt in range(ntypex):
            for tt2 in range(ntype):
                Gi = G[tt*ntype+tt2]
                si = s[tt]

                dG_ds_i = []
                for ii in range(M1):
                    dG_ds_ii = tf.reshape(tf.gradients(Gi[:,ii], si), [-1, 1])
                    dG_ds_i.append(dG_ds_ii)
                dG_ds_i = tf.concat(dG_ds_i, axis=1)
                dG_ds.append(dG_ds_i)
        return dG_ds
    
    def build_u2s_u2ds(self):
        dic_ph = {}
        dic_ph['u'] = tf.placeholder(tf.float32, [None, 1], 't_u')
        dic_ph['r2'] = self.build_u2r(dic_ph['u'])
        dic_ph['s'], dic_ph['sr'] = self.build_r2s(dic_ph['r2'])
        dic_ph['ds_dr2'], dic_ph['dsr_dr2'] = self.build_ds_dr(dic_ph['r2'], dic_ph['s'], dic_ph['sr'])

        return dic_ph

    def run_u2s(self):
        ntypex = nvnmd_cfg.dscp['ntypex']
        avg, std = get_normalize(nvnmd_cfg.weight)
        NBIT_FEA_X = nvnmd_cfg.nbit['NBIT_FEA_X']

        dic_ph = self.build_u2s_u2ds()
        sess = get_sess()

        N = 2 ** NBIT_FEA_X
        u = 1.0 * np.arange(0,N) / N
        u = np.reshape(u, [-1,1])
        feed_dic = {dic_ph['u']:u}
        key = 'u,r2,s,sr,ds_dr2,dsr_dr2'
        tlst = [dic_ph[k] for k in key.split(',')]
        res = sess.run(tlst, feed_dic)

        res2 = {}
        key = key.split(',')
        for ii in range(len(key)):
            res2[key[ii]] = res[ii]

        # change value
        # set 0 value, when u=0
        for tt in range(ntypex):
            res2['s'][tt][0] = -avg[tt,0] / std[tt,0]
            res2['sr'][tt][0] = 0
            res2['ds_dr2'][tt][0] = 0
            res2['dsr_dr2'][tt][0] = 0
        
        
        r = np.sqrt(res2['r2'])
        sess.close()

        return res2 
    
    def build_s2G_s2dG(self):
        ntype = nvnmd_cfg.dscp['ntype']
        dic_ph = {}
        dic_ph['s'] = [tf.placeholder(tf.float32, [None, 1], 't_s%d'%tt) for tt in range(ntype)]
        dic_ph['G'] = self.build_s2G(dic_ph['s'])
        dic_ph['dG_ds'] = self.build_dG_ds(dic_ph['G'], dic_ph['s'])
        return dic_ph
    
    def run_s2G(self, dat):

        dic_ph = self.build_s2G_s2dG()
        sess = get_sess()

        feed_dic = dict(zip(dic_ph['s'], dat['s']))
        key = 'G,dG_ds'
        tlst = [dic_ph[k] for k in key.split(',')]
        res = sess.run(tlst, feed_dic)

        res2 = {}
        key = key.split(',')
        for ii in range(len(key)):
            res2[key[ii]] = res[ii]
        
        sess.close()
        return res2 


from typing import List, Optional

def map(*, 
        nvnmd_config: Optional[str] = 'nvnmd/config.npy', 
        nvnmd_weight: Optional[str] = 'nvnmd/weight.npy', 
        nvnmd_map: Optional[str] = 'nvnmd/map.npy', 
        **kwargs
        ):
    mapObj = Map(nvnmd_config, nvnmd_weight, nvnmd_map)
    mapObj.build_map()
