import numpy as np 

from deepmd.nvnmd.data.data import jdata_deepmd_input
from deepmd.nvnmd.utils.config import nvnmd_cfg 
from deepmd.nvnmd.utils.fio import FioBin, FioDic, FioTxt, FioHead
from deepmd.nvnmd.utils.encode import Encode
from deepmd.nvnmd.utils.atoms import Atoms


class Vcs:

    def __init__(self,
        config_file: str="nvnmd/config.npy",
        debug_file: str="nvnmd/debug/res.npy",
        model_file: str="nvnmd/model.pb",
        vcs_path: str='nvnmd/vcs'
        ) -> None:
        self.config_file = config_file
        self.model_file = model_file 
        self.debug_file = debug_file
        self.vcs_path = vcs_path

        jdata = jdata_deepmd_input['nvnmd']
        jdata['config_file'] = config_file
        jdata['enable'] = True 

        nvnmd_cfg.init_from_jdata(jdata)
        self.debug = FioDic().load(debug_file, {})

    def gen_input(self):
        ncfg = nvnmd_cfg.nbit['NCFG']
        nnet = nvnmd_cfg.nbit['NNET']
        nfea = nvnmd_cfg.nbit['NFEA']
        NSEL = nvnmd_cfg.ctrl['NSEL']

        cfg, wfp, wbp, fea, gra = self.get_model(self.model_file) 
        e = Encode()

        # wfp
        head = self.get_head(ncfg=0, nnet=nnet, nfea=0, ngra=0, nlst=0, natm=0)
        swfp = [head]
        swfp.extend(wfp)
        e.extend_list(swfp, np.ceil(len(swfp)/32)*32) #防止仿真读溢出
        FioTxt().save(self.vcs_path+'/lmp1_hex.txt', swfp)
        head_info = FioHead().info()

        print(f"{head_info} : wfp {len(wfp)}")
        # wbp
        head = self.get_head(ncfg=0, nnet=nnet, nfea=0, ngra=0, nlst=0, natm=0)
        swbp = [head]
        swbp.extend(wbp)
        e.extend_list(swbp, np.ceil(len(swbp)/32)*32) #防止仿真读溢出
        FioTxt().save(self.vcs_path+'/lmp2_hex.txt', swbp)

        print(f"{head_info} : wbp {len(wbp)}")
        # wfp and wbp according to NSEL (parameter of Time division multiplexing)
        nmerge = nnet // NSEL 
        wfp = [wfp[ii][-72//4:] for ii in range(nnet)]
        swfp = [''.join(wfp[ii*nmerge:(ii+1)*nmerge][::-1]) for ii in range(NSEL)]
        FioTxt().save(self.vcs_path+'/wfp_hex.txt', swfp)

        wbp = [wbp[ii][-72//4:] for ii in range(nnet)]
        swbp = [''.join(wbp[ii*nmerge:(ii+1)*nmerge][::-1]) for ii in range(NSEL)]
        FioTxt().save(self.vcs_path+'/wbp_hex.txt', swbp)
        # data from cpu to fpga : cfg, fea, and gra
        head = self.get_head(ncfg=ncfg, nnet=0, nfea=nfea, ngra=nfea, nlst=0, natm=0)
        scfg = [head]
        scfg = scfg + cfg + fea + gra 
        e.extend_list(scfg, np.ceil(len(scfg)/32)*32) #防止仿真读溢出
        FioTxt().save(self.vcs_path+'/lmp3_hex.txt', scfg)

        print(f"{head_info} : cfg {len(cfg)}")
        print(f"{head_info} : fea {len(fea)}")
        print(f"{head_info} : gra {len(gra)}")

        # nlst and atm
        nlst, spes, crds = self.rebuild_atoms()
        hlst = self.get_lst(nlst)
        hatm = self.get_atm(spes, crds)
        head = self.get_head(ncfg=0, nnet=0, nfea=0, ngra=0, nlst=len(hlst), natm=len(hatm))
        sla = [head]
        sla = sla + hlst + hatm 
        e.extend_list(sla, np.ceil(len(sla)/32)*32) #防止仿真读溢出
        FioTxt().save(self.vcs_path+'/lmp4_hex.txt', sla)

        print(f"{head_info} : nlst {len(nlst)}")
        print(f"{head_info} : crds {len(crds)}")
        # force
        hfor = ['0']*len(crds)
        FioTxt().save(self.vcs_path+'/for_hex.txt', hfor)


    def get_lst(self, nlst):
        NI = nvnmd_cfg.dscp['NI']
        NBIT_LST_MAX = nvnmd_cfg.nbit['NBIT_LST_MAX']

        natom = np.size(nlst) // NI
        e = Encode()

        # lst 
        nlst = np.reshape(nlst, [natom, -1])
        for ii in range(natom):
            v = nlst[ii]
            v[v == -1] = ii 
            nlst[ii] = v 
        nlpl = 512 // NBIT_LST_MAX # number of nlst per line
        nl = NI // nlpl # number of line per atom
        nlst = np.reshape(nlst, [natom * nl , -1])
        bnlst = e.dec2bin(nlst, NBIT_LST_MAX, False)
        bnlst = e.reverse_bin(bnlst, nlpl)
        bnlst = e.merge_bin(bnlst, nlpl)
        hnlst = e.bin2hex(bnlst)

        return hnlst

    def get_atm(self, spes, crds):
        NBIT_LONG_DATA_FL = nvnmd_cfg.nbit['NBIT_LONG_DATA_FL']
        NBIT_LONG_DATA = nvnmd_cfg.nbit['NBIT_LONG_DATA']

        natom = len(spes)
        e = Encode()

        # atm
        spes = np.reshape(spes, [natom, 1])
        crds = np.reshape(crds, [natom, 3])
        crds = e.qr(crds, NBIT_LONG_DATA_FL)
        atms = np.zeros([natom, 4])
        atms[:,0:3] = crds 
        atms[:,3:4] = spes 
        batm = e.dec2bin(atms, NBIT_LONG_DATA, True)
        batm = e.extend_bin(batm, 64)
        batm = e.reverse_bin(batm, 8)
        batm = e.merge_bin(batm, 8)
        hatm = e.bin2hex(batm)

        return hatm
    
    def get_model(self, model_file):
        ncfg = nvnmd_cfg.nbit['NCFG']
        nnet = nvnmd_cfg.nbit['NNET']
        nfea = nvnmd_cfg.nbit['NFEA']

        model_bin = FioBin().load(model_file, b'')
        model_hex = model_bin.hex()
        nhex = 512 // 4 
        model = [model_hex[ii*nhex:(ii+1)*nhex] for ii in range(len(model_hex)//nhex)]

        st = 0
        dt = 1; head = model[st:st+dt]; st += dt  
        dt = ncfg; cfg = model[st:st+dt]; st += dt  
        dt = nnet; wfp = model[st:st+dt]; st += dt 
        dt = nnet; wbp = model[st:st+dt]; st += dt 
        dt = nfea; fea = model[st:st+dt]; st += dt 
        dt = nfea; gra = model[st:st+dt]; st += dt 
        return cfg, wfp, wbp, fea, gra 
    
    def get_head(self, ncfg=0, nnet=0, nfea=0, ngra=0, nlst=0, natm=0):
        """: generate the head line of one frame data
        """
        prec = 2**16 
        stype = (1 if(ncfg>0) else 0) | (2 if(nnet>0) else 0) | (4 if(nfea>0) else 0) | \
                (8 if(ngra>0) else 0) | (16 if(nlst>0) else 0) | (32 if(natm>0) else 0)
        stype = hex(prec+stype)[-4:]

        check_ed = hex(prec + (int('f0f0', 16) ^ int('0001', 16) ^ int(stype, 16)))[-4:]

        ncfg = hex(prec+ncfg)[-4:]
        nnet = hex(prec+nnet)[-4:]
        nfea = hex(prec+nfea)[-4:]
        ngra = hex(prec+ngra)[-4:]
        nlst = hex(prec+nlst)[-4:]
        natm = hex(prec+natm)[-4:]

        head = "0f0f_%s_%s_%s_%s_%s_%s_%s_%s_0001_f0f0"%(check_ed, natm, nlst, ngra, nfea, nnet, ncfg, stype)
        return head.replace('_','')
    
    def rebuild_atoms(self):
        nlst = self.debug['o_nlist']
        rij = self.debug['rij']
        spes = self.debug['t_type']
        crds = self.debug['t_coord']
        box = self.debug['t_box']
        return Atoms().extend(nlst, rij, spes, crds, box)
    
def vcs(*,
        nvnmd_config: str = 'nvnmd/config.npy', 
        nvnmd_debug: str = 'nvnmd/debug', 
        nvnmd_model: str = 'nvnmd/model.pb', 
        nvnmd_vcs: str = 'nvnmd/vcs', 
        **kwargs
        ):
        vcsObj = Vcs(
            nvnmd_config, 
            nvnmd_debug,
            nvnmd_model, 
            nvnmd_vcs
            )
        vcsObj.gen_input() 
        

