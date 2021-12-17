
import numpy as np

from deepmd.nvnmd.utils.fio import FioHead, FioBin, FioTxt
from deepmd.nvnmd.utils.config import nvnmd_cfg
from deepmd.nvnmd.utils.weight import get_fitnet_weight, get_r2u_u2r_param 
from deepmd.nvnmd.utils.encode import Encode 

from deepmd.nvnmd.data.data import jdata_deepmd_input

class Wrap():

    def __init__(self,
        config_file: str,
        weight_file: str,
        map_file: str,
        model_file: str
        ) -> None:
        self.config_file = config_file
        self.weight_file = weight_file
        self.map_file = map_file 
        self.model_file = model_file

        jdata = jdata_deepmd_input['nvnmd']
        jdata['config_file'] = config_file
        jdata['weight_file'] = weight_file
        jdata['map_file'] = map_file
        jdata['enable'] = True 

        nvnmd_cfg.init_from_jdata(jdata)
    
    def wrap(self):
        # head = FioHead().info()
        # print(f"{head}: star to wrap model")

        dscp = nvnmd_cfg.dscp 
        ctrl = nvnmd_cfg.ctrl 

        M1 = dscp['M1']
        ntype = dscp['ntype']
        ntype_max = dscp['ntype_max']
        NSTDM_M1X = ctrl['NSTDM_M1X']
        e = Encode()

        bcfg = self.wrap_dscp()
        bfps, bbps = self.wrap_fitn()
        bfea, bgra = self.wrap_map()

        # split data with {nbit} bits per row 
        hcfg = e.bin2hex(e.split_bin(bcfg, 64))

        hfps = e.bin2hex(e.split_bin(bfps, 72))
        # hfps = e.extend_list(hfps, (len(hfps) // ntype) * ntype_max)

        hbps = e.bin2hex(e.split_bin(bbps, 72))
        # hbps = e.extend_list(hbps, (len(hbps) // ntype) * ntype_max)

        # split into multiple rows
        bfea = e.split_bin(bfea, len(bfea[0]) // NSTDM_M1X)
        # bfea = e.reverse_bin(bfea, NSTDM_M1X)
        # extend the number of lines
        hfea = e.bin2hex(bfea)
        hfea = e.extend_list(hfea, (len(hfea) // ntype) * ntype_max)

        # split into multiple rows
        bgra = e.split_bin(bgra, len(bgra[0]) // NSTDM_M1X)
        # bgra = e.reverse_bin(bgra, NSTDM_M1X)
        # extend the number of lines
        hgra = e.bin2hex(bgra)
        hgra = e.extend_list(hgra, (len(hgra) // ntype) * ntype_max)

        # extend data according to the number of bits per row of BRAM
        nhex = 512
        hcfg = e.extend_hex(hcfg, nhex)
        hfps = e.extend_hex(hfps, nhex)
        hbps = e.extend_hex(hbps, nhex)
        hfea = e.extend_hex(hfea, nhex)
        hgra = e.extend_hex(hgra, nhex)

        head = FioHead().info()
        print("%s: len(hcfg) : %d"%(head, len(hcfg)))
        print("%s: len(hfps) : %d"%(head, len(hfps)))
        print("%s: len(hbps) : %d"%(head, len(hbps)))
        print("%s: len(hfea) : %d"%(head, len(hfea)))
        print("%s: len(hgra) : %d"%(head, len(hgra)))

        # FioTxt().save('nvnmd/wrap/hcfg.txt', hcfg)
        # FioTxt().save('nvnmd/wrap/hfps.txt', hfps)
        # FioTxt().save('nvnmd/wrap/hbps.txt', hbps)
        # FioTxt().save('nvnmd/wrap/hfea.txt', hfea)
        # FioTxt().save('nvnmd/wrap/hgra.txt', hgra)
        #
        hs = []
        hs.extend(hcfg)
        hs.extend(hfps)
        hs.extend(hbps)
        hs.extend(hfea)
        hs.extend(hgra)

        FioBin().save(self.model_file, hs)

    
    def wrap_dscp(self):
        # head = FioHead().info()
        # print(f"{head}: star to wrap model/dscp")

        dscp = nvnmd_cfg.dscp 
        nbit = nvnmd_cfg.nbit 
        maps = nvnmd_cfg.map
        NBIT_DATA = nbit['NBIT_DATA']
        NBIT_DATA_FL = nbit['NBIT_DATA_FL']
        NBIT_FEA_FL = nbit['NBIT_FEA_FL']
        NBIT_LST = nbit['NBIT_LST']
        NBIT_SHIFT = nbit['NBIT_SHIFT']
        
        bs = ''
        e = Encode()
        # r2u
        rcut = dscp['rcut']
        _xps, _ks, _bs, _xps2, _ks2, _bs2 = get_r2u_u2r_param(rcut)
        R2U_XS = _xps[:,0]
        R2U_XS = np.append(R2U_XS, rcut**2)
        R2U_XS = e.qr(R2U_XS, NBIT_DATA_FL)
        R2U_KS = -np.round(np.log2(_ks))
        R2U_BS = e.qr(_bs, NBIT_DATA_FL)

        bs = e.dec2bin(R2U_XS[0], NBIT_DATA)[0] + bs 
        bs = e.dec2bin(R2U_XS[1], NBIT_DATA)[0] + bs 
        bs = e.dec2bin(R2U_XS[2], NBIT_DATA)[0] + bs 
        bs = e.dec2bin(R2U_KS[0], NBIT_DATA)[0] + bs 
        bs = e.dec2bin(R2U_KS[1], NBIT_DATA)[0] + bs 
        bs = e.dec2bin(R2U_KS[2], NBIT_DATA)[0] + bs 
        bs = e.dec2bin(R2U_BS[0], NBIT_DATA)[0] + bs 
        bs = e.dec2bin(R2U_BS[1], NBIT_DATA)[0] + bs 
        bs = e.dec2bin(R2U_BS[2], NBIT_DATA)[0] + bs 
        # sel
        SEL = dscp['SEL']
        bs = e.dec2bin(SEL[0], NBIT_LST)[0] + bs 
        bs = e.dec2bin(SEL[1], NBIT_LST)[0] + bs 
        bs = e.dec2bin(SEL[2], NBIT_LST)[0] + bs 
        bs = e.dec2bin(SEL[3], NBIT_LST)[0] + bs 
        # G*s
        ntypex = dscp['ntypex']
        ntype = dscp['ntype']
        ntypex_max = dscp['ntypex_max']
        ntype_max = dscp['ntype_max']
        M1 = dscp['M1']
        GSs = []
        for tt in range(ntypex_max):
            for tt2 in range(ntype_max):
                if (tt<ntypex) and (tt2<ntype):
                    s = maps[f's_t{tt}_t{tt2}']
                    G = maps[f'G_t{tt}_t{tt2}']
                    v = s[0] * G[0,:]
                else:
                    v = np.zeros(M1)
                for ii in range(M1):
                    GSs.extend(e.dec2bin(e.qr(v[ii], 2*NBIT_FEA_FL), 27, True))
        sGSs = ''.join(GSs[::-1])
        bs = sGSs + bs 
        #
        NIX = dscp['NIX']
        ln2_NIX = int(np.log2(NIX))
        bs = e.dec2bin(ln2_NIX, NBIT_SHIFT)[0] + bs 
        return bs 

    def wrap_fitn(self):
        """: wrap the weights of fitting net
        """
        # head = FioHead().info()
        # print(f"{head}: star to wrap model/fitn")

        dscp = nvnmd_cfg.dscp 
        fitn = nvnmd_cfg.fitn 
        weight = nvnmd_cfg.weight
        nbit = nvnmd_cfg.nbit 
        ctrl = nvnmd_cfg.ctrl

        ntype = dscp['ntype'] 
        ntype_max = dscp['ntype_max'] 
        nlayer_fit = fitn['nlayer_fit'] 
        NNODE_FITS = fitn['NNODE_FITS']
        NBIT_SUM = nbit['NBIT_SUM']
        NBIT_DATA_FL = nbit['NBIT_DATA_FL']
        NUM_WLN2 = nbit['NUM_WLN2']
        NBIT_WLN2 = nbit['NBIT_WLN2']
        NBIT_SPE = nbit['NBIT_SPE']
        NSTDM = ctrl['NSTDM']
        NSEL = ctrl['NSEL']


        # encode all parameters
        bb, bs, bw = [], [], []
        for ll in range(nlayer_fit):
            bbt, bst, bwt = [], [], [] 
            for tt in range(ntype_max):
                # get parameters: weight and bias
                if (tt < ntype):
                    w, b = get_fitnet_weight(weight, tt, ll, nlayer_fit)
                else:
                    w, b = get_fitnet_weight(weight, 0, ll, nlayer_fit)
                    w = w * 0
                    b = b * 0
                # restrict the shift value of energy
                if (ll == (nlayer_fit-1)):
                    b = b * 0 
                bbi = self.wrap_bias(b, NBIT_SUM, NBIT_DATA_FL)
                bsi, bwi = self.wrap_weight(w, NUM_WLN2, NBIT_DATA_FL, NBIT_SPE, NBIT_WLN2)
                bbt.append(bbi)
                bst.append(bsi)
                bwt.append(bwi)
            bb.append(bbt)
            bs.append(bst)
            bw.append(bwt)
        #
        bfps, bbps = [], []
        for ss in range(NSEL):
            tt = ss // NSTDM
            sc = ss % NSTDM 
            sr = ss % NSTDM 
            bfp, bbp = '', ''
            for ll in range(nlayer_fit):
                nr = NNODE_FITS[ll]
                nc = NNODE_FITS[ll+1]
                nrs = int(np.ceil(nr/NSTDM))
                ncs = int(np.ceil(nc/NSTDM))
                nw = NUM_WLN2
                if (nc == 1):
                    #final layer
                    #* fp *#
                    bi = [bs[ll][tt][sr*nrs+rr][cc][ww] for rr in range(nrs) for cc in range(nc) for ww in range(nw)]; bi.reverse(); bfp = ''.join(bi) + bfp
                    bi = [bw[ll][tt][sr*nrs+rr][cc][ww] for rr in range(nrs) for cc in range(nc) for ww in range(nw)]; bi.reverse(); bfp = ''.join(bi) + bfp
                    bi = [bb[ll][tt][sc*ncs*0+cc] for cc in range(ncs)]; bi.reverse(); bfp = ''.join(bi) + bfp
                    #* bp *#
                    bi = [bs[ll][tt][sr*nrs+rr][cc][ww] for rr in range(nrs) for cc in range(nc) for ww in range(nw)]; bi.reverse(); bbp = ''.join(bi) + bbp
                    bi = [bw[ll][tt][sr*nrs+rr][cc][ww] for rr in range(nrs) for cc in range(nc) for ww in range(nw)]; bi.reverse(); bbp = ''.join(bi) + bbp
                    bi = [bb[ll][tt][sc*ncs*0+cc] for cc in range(ncs)]; bi.reverse(); bbp = ''.join(bi) + bbp
                else:
                    #* fp *#
                    bi = [bs[ll][tt][rr][sc*ncs+cc][ww] for cc in range(ncs) for rr in range(nr) for ww in range(nw)]; bi.reverse(); bfp = ''.join(bi) + bfp
                    bi = [bw[ll][tt][rr][sc*ncs+cc][ww] for cc in range(ncs) for rr in range(nr) for ww in range(nw)]; bi.reverse(); bfp = ''.join(bi) + bfp
                    bi = [bb[ll][tt][sc*ncs+cc] for cc in range(ncs)]; bi.reverse(); bfp = ''.join(bi) + bfp
                    #* bp *#
                    bi = [bs[ll][tt][sr*nrs+rr][cc][ww] for rr in range(nrs) for cc in range(nc) for ww in range(nw)]; bi.reverse(); bbp = ''.join(bi) + bbp
                    bi = [bw[ll][tt][sr*nrs+rr][cc][ww] for rr in range(nrs) for cc in range(nc) for ww in range(nw)]; bi.reverse(); bbp = ''.join(bi) + bbp
                    bi = [bb[ll][tt][sc*ncs+cc] for cc in range(ncs)]; bi.reverse(); bbp = ''.join(bi) + bbp
            bfps.append(bfp)
            bbps.append(bbp)
        print(len(bfps), len(bfps[0]))
        return bfps, bbps 
                     
    
    def wrap_bias(self, bias, NBIT_SUM, NBIT_DATA_FL):
        # head = FioHead().info()
        # print(f"{head}: star to wrap model/fitn/bias")

        e = Encode()
        bias = e.qr(bias, NBIT_DATA_FL)
        Bs = e.dec2bin(bias, NBIT_SUM, True)
        return Bs 
    
    def wrap_weight(self, weight, NUM_WLN2, NBIT_DATA_FL, NBIT_SPE, NBIT_WLN2):
        # head = FioHead().info()
        # print(f"{head}: star to wrap model/fitn/weight")

        def pow_2n(xv):
            v = np.log2(1.5)
            eps = 1e-12
            xlog = np.log2(np.abs(xv) + eps)
            n = np.ceil(xlog - v)
            v2n = np.power(2.0, n)
            return v2n, n
        
        th = 2**(-NBIT_DATA_FL)
        sh = weight.shape
        nr = sh[0]
        nc = sh[1]
        signs = np.zeros([nr * nc, NUM_WLN2])
        w2n   = np.zeros([nr * nc, NUM_WLN2])

        # w => s1*2^n1 + s2*2^n2 + ... sn*2^nn
        # s > 0编码为1
        # s < 0编码为-1
        # |vw| < prec, s编码为0
        vw = weight.reshape([-1]).copy()
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
                #  - 0 + : 00 10 01
                arrs[ii,jj] = 2 if (signs[ii, jj] == 0) else (1 if (signs[ii, jj] == 1) else 0)
                # arrs[ii,jj] = 3 if signs[ii, jj] < 0 else signs[ii, jj]
                arrw[ii,jj] = NBIT_DATA_FL + w2n[ii, jj]
        
        # 编码为二进制
        arrs = np.reshape(arrs, [nr, nc, NUM_WLN2])
        arrw = np.reshape(arrw, [nr, nc, NUM_WLN2])

        e = Encode()
        Ss = [[[e.dec2bin(arrs[rr, cc, ww], NBIT_SPE )[0] for ww in range(NUM_WLN2)] for cc in range(nc)] for rr in range(nr)]
        Ws = [[[e.dec2bin(arrw[rr, cc, ww], NBIT_WLN2)[0] for ww in range(NUM_WLN2)] for cc in range(nc)] for rr in range(nr)]
        return Ss, Ws 

    def wrap_map(self):
        # head = FioHead().info()
        # print(f"{head}: star to wrap model/map")

        dscp = nvnmd_cfg.dscp
        maps = nvnmd_cfg.map 
        nbit = nvnmd_cfg.nbit 

        M1 = dscp['M1']
        ntype = dscp['ntype'] 
        NBIT_FEA = nbit['NBIT_FEA']
        NBIT_FEA_FL = nbit['NBIT_FEA_FL']

        keys = 's,sr,G'.split(',')
        keys2 = 'ds_dr2,dsr_dr2,dG_dr2'.split(',')

        e = Encode()

        datas = {}
        idxs = [[0,tt] for tt in range(ntype)]
        for ii in range(len(idxs)):
            tt, tt2 = idxs[ii]
            postfix = f'_t{tt}_t{tt2}'
            for key in (keys+keys2):
                if ii == 0: datas[key] = []
                datas[key].append(maps[key+postfix])
        
        for key in (keys+keys2):
            datas[key] = np.vstack(datas[key])
            datas[key] = e.qr(datas[key], NBIT_FEA_FL)
        # fea
        dat = [datas[key] for key in keys]
        dat = np.hstack(dat)
        dat = dat[:,::-1]
        bs = e.dec2bin(dat, NBIT_FEA, True)
        bs = e.merge_bin(bs, M1+2)
        bfea = bs
        # gra
        dat = [datas[key] for key in keys2]
        dat = np.hstack(dat)
        dat = dat[:,::-1]
        bs = e.dec2bin(dat, NBIT_FEA, True)
        bs = e.merge_bin(bs, M1+2)
        bgra = bs 
        return bfea, bgra 

        
from typing import List, Optional

def wrap(*, 
        nvnmd_config: Optional[str] = 'nvnmd/config.npy', 
        nvnmd_weight: Optional[str] = 'nvnmd/weight.npy', 
        nvnmd_map: Optional[str] = 'nvnmd/map.npy', 
        nvnmd_model: Optional[str] = 'nvnmd/model.pb', 
        **kwargs
        ):
    wrapObj = Wrap(nvnmd_config, nvnmd_weight, nvnmd_map, nvnmd_model)
    wrapObj.wrap()
