import numpy as np

from deepmd.env import tf
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION

from deepmd.utils.argcheck import normalize
from deepmd.utils.compat import updata_deepmd_input
from deepmd.train.run_options import RunOptions
from deepmd.train.trainer import DPTrainer
from deepmd.nvnmd.utils.fio import FioTxt, FioDic, FioHead, FioArrInt

from deepmd.nvnmd.utils.config import nvnmd_cfg
from deepmd.nvnmd.utils.atoms import Atoms
from deepmd.nvnmd.utils.encode import Encode

from deepmd.nvnmd.data.data import jdata_deepmd_input


class Debug:

    def __init__(self,
        config_file: str,
        weight_file: str,
        map_file: str,
        output_path: str,
        atoms_file: str,
        type_map_file: str
        ) -> None:
        self.config_file = config_file
        self.weight_file = weight_file
        self.map_file = map_file 
        self.output_path = output_path
        self.atoms_file = atoms_file
        self.type_map_file = type_map_file
        # nvnmd
        jdata = jdata_deepmd_input['nvnmd']
        jdata['config_file'] = config_file
        jdata['weight_file'] = weight_file
        jdata['map_file'] = map_file
        jdata['enable'] = True 
        jdata['restore_descriptor'] = True 
        jdata['restore_fitting_net'] = True 
        jdata['quantize_descriptor'] = True 
        jdata['quantize_fitting_net'] = True 

        nvnmd_cfg.init_from_jdata(jdata)
        # atoms
        self.type_map = FioTxt().load(self.type_map_file)
        self.atoms = Atoms().load(self.atoms_file)
        self.atoms = Atoms().resort_by_type(self.atoms, self.type_map)
        # config
        nvnmd_cfg.save('nvnmd/debug/config.npy')

    def init_model(self):
        self.sess = tf.Session()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
    
    def build(self):
        """:
        reference: deepmd/entrypoints/train.py
                   deepmd/train/trainer.py
        """
        ntype = nvnmd_cfg.dscp['ntype']
        jdata = nvnmd_cfg.get_deepmd_jdata()
        run_opt = RunOptions(log_path=None, log_level=20)
        jdata = updata_deepmd_input(jdata, warning=True, dump="input_v2_compat.json")
        jdata = normalize(jdata)
        self.trainer = DPTrainer(jdata, run_opt, False)
        self.model = self.trainer.model
        # place holder
        self.place_holders = {}
        self.place_holders['coord'] = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], 't_coord')
        self.place_holders['box'] = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], 't_box')
        self.place_holders['type']      = tf.placeholder(tf.int32,   [None], name='t_type')
        self.place_holders['natoms_vec']        = tf.placeholder(tf.int32,   [ntype+2], name='t_natoms')
        self.place_holders['default_mesh']      = tf.placeholder(tf.int32,   [None], name='t_mesh')
        self.place_holders['is_training']       = tf.placeholder(tf.bool)
        #
        # for key in self.place_holders.keys():
        #     print(f"\033[1;32;48m {key} \033[0m :", self.place_holders[key])
        # build model
        self.model_pred\
            = self.model.build (self.place_holders['coord'], 
                                self.place_holders['type'], 
                                self.place_holders['natoms_vec'], 
                                self.place_holders['box'], 
                                self.place_holders['default_mesh'],
                                self.place_holders,
                                frz_model=None,
                                suffix = "", 
                                reuse = False)
    
    def get_feed_dic(self, atoms, dic_ph):
        ntype = nvnmd_cfg.dscp['ntype']
        #
        natom = len(atoms)
        spe = atoms.get_atomic_numbers()
        type_map_an = Atoms().spe2atn(self.type_map)
        #
        natoms_vec_dat = [natom, natom]
        for tt in range(ntype):
            n = np.sum(spe==type_map_an[tt])
            natoms_vec_dat.append(n)
        natoms_vec_dat = np.int32(np.array(natoms_vec_dat))
        #
        type_dat = []
        for tt in range(ntype): 
            type_dat = type_dat + [tt] * natoms_vec_dat[2+tt]
        #
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
        #
        self.natom = natom 
        self.natom_vec = natoms_vec_dat[2:]
        return feed_dic
    
    def get_MapNvnmd_name(self, name):
        name_fp = f'{name}/MapNvnmd'
        name_bp = f'gradients/{name_fp}_grad/Reshape'
        return name_fp+':0', name_bp+':0'

    def get_QuantizeNvnmd_name(self, name):
        name_fp = f'{name}/QuantizeNvnmd'
        name_bp = f'gradients/{name_fp}_grad/QuantizeNvnmd'
        return name_fp+':0', name_bp+':0'
    
    def get_MatmulNvnmd_name(self, name):
        name_fp = f'{name}/MatmulNvnmd'
        name_bp = f'gradients/{name_fp}_grad/MatmulNvnmd'
        return name_fp+':0', name_bp+':0'

    def get_Tanh2Nvnmd_name(self, name):
        name_fp = f'{name}/Tanh2Nvnmd'
        name_bp = f'gradients/{name_fp}_grad/add_1'
        return name_fp+':0', name_bp+':0'
    
    def get_Add_name(self, name):
        name_fp = f'{name}/add'
        name_bp = f'gradients/{name_fp}_grad/Reshape'
        return name_fp+':0', name_bp+':0'
    
    def get_name_idx(self, name, idx):
        if idx == 0:
            return name 
        else:
            return f'{name}_{idx}'
    
    def get_dscp_namedic(self):
        ntypex = nvnmd_cfg.dscp['ntypex']
        ntype = nvnmd_cfg.dscp['ntype']

        namedic = {}

        keys = 'u,rij,s,sr,Rxyz'.split(',')
        for tt in range(ntypex):
            for tt2 in range(ntype):
                for key in keys:
                    name = f'filter_type_all_x/{key}'
                    key2 = f'{key}'
                    namedic[key2], namedic[f'grad_{key2}'] = self.get_QuantizeNvnmd_name(name)

        keys = 'G'.split(',')
        for tt in range(ntypex):
            for tt2 in range(ntype):
                for key in keys:
                    name = f'filter_type_all/{key}'
                    name = self.get_name_idx(name, tt2)
                    key2 = f'{key}_t{tt}_t{tt2}'
                    namedic[key2], namedic[f'grad_{key2}'] = self.get_QuantizeNvnmd_name(name)
        
        keys = 'GR,d'.split(',')
        for key in keys:
            name = f'filter_type_all/{key}'
            key2 = f'{key}'
            namedic[key2], namedic[f'grad_{key2}'] = self.get_QuantizeNvnmd_name(name)
        
        return namedic
    
    def get_fitn_namedic(self):
        ntype = nvnmd_cfg.dscp['ntype']
        nlayer_fit = nvnmd_cfg.fitn['nlayer_fit']
        namedic = {}
        for tt in range(ntype):
            for ll in range(nlayer_fit):
                is_final = ll == (nlayer_fit-1)
                head = f"final_layer_type_{tt}" if is_final else f"layer_{ll}_type_{tt}"
                # wx
                name = f"{head}/wx"
                key2 = f"wx_l{ll}_t{tt}"
                namedic[key2], namedic[f'grad_{key2}'] = self.get_MatmulNvnmd_name(name)
                # wxb
                name = f"{head}/wxb"
                key2 = f"wxb_l{ll}_t{tt}"
                namedic[key2], namedic[f'grad_{key2}'] = self.get_Add_name(name)
                # actfun
                name = f"{head}/actfun"
                key2 = f"actfun_l{ll}_t{tt}"
                if is_final:
                    namedic[key2], namedic[f'grad_{key2}'] = self.get_Add_name(name)
                else:
                    namedic[key2], namedic[f'grad_{key2}'] = self.get_Tanh2Nvnmd_name(name)
        return namedic

    def get_tensor_names(self):
        names = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        names2 = []
        for name in names:
            if (name.startswith('t_')) or (name.startswith('o_')):
                names2.append(name)
        return names2 

    def get_namedic(self):
        # all name of tensor
        # names = self.get_tensor_names()
        # print(names)

        # input and output
        keys1 = 't_coord,t_box,t_type,t_natoms,t_mesh'.split(',')
        keys3 = 'o_rmat,o_rmat_deriv,o_rij,o_nlist,o_descriptor'.split(',')
        keys2 = 'o_atom_energy,o_energy,o_force,o_virial,o_atom_virial'.split(',')
        keys = keys1 + keys2 + keys3 
        
        namedic = {}
        for key in keys:
            namedic[key] = f"{key}:0"
        
        namedic.update(self.get_dscp_namedic())
        namedic.update(self.get_fitn_namedic())

        return namedic
    
    def get_tensordic(self, namedic):
        tensordic = {}
        graph = tf.get_default_graph()
        for key in namedic.keys():
            tensordic[key] = graph.get_tensor_by_name(namedic[key])
        return tensordic
    
    def get_valuedic(self, sess, feed_dict, tensordic):
        valuelist = sess.run(list(tensordic.values()), feed_dict=feed_dict)
        valuedic = dict(zip(tensordic.keys(), valuelist))
        return valuedic
    
    def merge_type_type(self, keys, valuedic):
        ntype = nvnmd_cfg.dscp['ntype']
        ntypex = nvnmd_cfg.dscp['ntypex']
        sel = nvnmd_cfg.dscp['sel']
        natom = self.natom 

        valuedic2 = {}
        valuedic2.update(valuedic)
        for key in keys:
            key_grad = f'grad_{key}'
            valuedic2[key] = []
            valuedic2[key_grad] = []
            for tt in range(ntypex):
                for tt2 in range(ntype):
                    sel_t = sel[tt2]
                    key2 = f'{key}_t{tt}_t{tt2}'
                    key2_grad = f'grad_{key2}'
                    value = valuedic[key2]
                    value_grad = valuedic[key2_grad]
                    valuedic2[key].append(np.reshape(value, [natom, sel_t,-1]))
                    valuedic2[key_grad].append(np.reshape(value_grad, [natom, sel_t,-1]))
                    del valuedic2[key2]
                    del valuedic2[key2_grad]
            valuedic2[key] = np.concatenate(valuedic2[key], axis=1)
            valuedic2[key_grad] = np.concatenate(valuedic2[key_grad], axis=1)
        
        return valuedic2
    
    def merge_type(self, keys, valuedic):
        ntype = nvnmd_cfg.dscp['ntype']
        natom_vec = self.natom_vec

        valuedic2 = {}
        valuedic2.update(valuedic)
        for key in keys:
            key_grad = f'grad_{key}'
            valuedic2[key] = []
            valuedic2[key_grad] = [] 
            for tt in range(ntype):
                natomi = natom_vec[tt]
                key2 = f'{key}_t{tt}'
                key2_grad = f'grad_{key2}'
                value = valuedic[key2]
                value_grad = valuedic[key2_grad]
                valuedic2[key].append(np.reshape(value, [natomi,-1]))
                valuedic2[key_grad].append(np.reshape(value_grad, [natomi, -1]))
                del valuedic2[key2]
                del valuedic2[key2_grad]
            
            valuedic2[key] = np.concatenate(valuedic2[key], axis=0)
            valuedic2[key_grad] = np.concatenate(valuedic2[key_grad], axis=0)
        return valuedic2
    
    def s1_merge(self, valuedic):
        keys = 'G'.split(',')
        valuedic = self.merge_type_type(keys, valuedic)

        nlayer_fit = nvnmd_cfg.fitn['nlayer_fit']
        keys = 'wx,wxb,actfun'.split(',')
        keys = [f'{key}_l{ll}' for key in keys for ll in range(nlayer_fit)]
        valuedic = self.merge_type(keys, valuedic)

        return valuedic 

    def resort_nlst(self, nlst):
        NI = nvnmd_cfg.dscp['NI']
        NIDP = nvnmd_cfg.dscp['NIDP']
        natom = self.natom

        nlst = np.reshape(nlst, [natom, -1])
        idx = np.zeros([natom, NI], dtype=np.int32) - 1

        for ii in range(natom):
            tt = 0
            for jj in range(NIDP):
                if nlst[ii, jj] != -1:
                    idx[ii, tt] = jj 
                    tt += 1 
        return idx 
    
    def sort_by_nlst(self, arr, idx_nlst, init_value=0):
        natom = self.natom 
        NI = nvnmd_cfg.dscp['NI']
        NIDP = nvnmd_cfg.dscp['NIDP']

        arr = np.reshape(arr, [natom, NIDP, -1])
        ndim = arr.shape[2]
        if ndim == 1:
            arr2 = np.zeros([natom, NI]) + init_value 
        else:
            arr2 = np.zeros([natom, NI, ndim]) + init_value 

        for ii in range(natom):
            for jj in range(NI):
                jj2 = idx_nlst[ii,jj]
                if jj2 != -1:
                    arr2[ii, jj] = arr[ii, jj2]
        return arr2 

    def s2_resort(self, valuedic):
        natom = self.natom 
        NI = nvnmd_cfg.dscp['NI']
        NIDP = nvnmd_cfg.dscp['NIDP']

        nlst = valuedic['o_nlist']
        idx_nlst = self.resort_nlst(nlst)

        for key in valuedic.keys():
            value = valuedic[key]
            if np.size(value) % int(natom * NIDP) == 0:
                init_value = 0
                if (key in ['o_nlist']): init_value = -1
                valuedic[key] = self.sort_by_nlst(value, idx_nlst, init_value)
        return valuedic 
    
    def s3_save(self, valuedic):
        self.save_input(valuedic) 
        self.save_output(valuedic) 
        self.save_dscp(valuedic) 
        self.save_fitn(valuedic) 
        self.save_cal_fij(valuedic)
        FioDic().save('nvnmd/debug/res.npy', valuedic)
    
    def check_range(self, value, name):
        vmin = int(np.min(value))
        vmax = int(np.max(value))
        vmax_abs = int(np.max(np.abs(value)))
        n = int(np.ceil(np.log2(vmax_abs+1)))
        # print(f"# {name} : {vmin} {vmax} {vmax_abs} {n}")
        print(f" {name:20} : {vmin:15} {vmax:15} {vmax_abs:15} {n:10}")

    def save_value(self, name, value, nbit):
        self.check_range(value, name)
        FioArrInt().save(f"{self.output_path}/{name}.txt", value, nbit)

    def save_input(self, valuedic):
        nbit = nvnmd_cfg.nbit 
        NBIT_DATA_FL = nbit['NBIT_DATA_FL']
        NBIT_DATA = nbit['NBIT_DATA']
        NBIT_LONG_DATA_FL = nbit['NBIT_LONG_DATA_FL']
        NBIT_LONG_DATA = nbit['NBIT_LONG_DATA']
        e = Encode()
        # t_coord,t_box,t_type,t_natoms,t_mesh
        # value = e.qr(valuedic['t_coord'], NBIT_LONG_DATA_FL)
        # self.save_value('coord',value, NBIT_LONG_DATA)
    
    def save_output(self, valuedic):
        nbit = nvnmd_cfg.nbit 
        NBIT_LST = nbit['NBIT_LST']
        NBIT_DATA_FL = nbit['NBIT_DATA_FL']
        e = Encode()
        # o_rmat,o_rmat_deriv,o_rij,o_nlist,o_descriptor
        value = valuedic['o_nlist']
        self.save_value('nlst', value, NBIT_LST)
        # o_atom_energy,o_energy,o_force,o_virial,o_atom_virial
        value = e.qr(valuedic['o_force'].reshape([-1, 3]), 2*NBIT_DATA_FL-1)
        self.save_value('fij', value, 32) 

    def save_dscp(self, valuedic):
        natom = self.natom 
        nbit = nvnmd_cfg.nbit 
        NBIT_FEA = nbit['NBIT_FEA']
        NBIT_FEA_FL = nbit['NBIT_FEA_FL']
        NBIT_DATA = nbit['NBIT_DATA']
        NBIT_DATA_FL = nbit['NBIT_DATA_FL']
        NBIT_DATA_FEA = nbit['NBIT_DATA_FEA']
        NBIT_DATA_FEA_FL = nbit['NBIT_DATA_FEA_FL']
        NIX = nvnmd_cfg.dscp['NIX']
        M1 = nvnmd_cfg.dscp['M1']

        e = Encode()
        # u,rij,s,sr,G,Rxyz,GR,d
        value = e.qr(valuedic['u'], NBIT_DATA_FL)
        self.save_value('u', value, NBIT_DATA)
        value = e.qr(valuedic['grad_u'], NBIT_DATA_FEA_FL)
        self.save_value('grad_u', value, NBIT_DATA_FEA)
        #
        value = e.qr(valuedic['rij'], NBIT_DATA_FL)
        self.save_value('rij', value, NBIT_DATA)
        #
        value = e.qr(valuedic['s'], NBIT_FEA_FL)
        self.save_value('s', value, NBIT_FEA)
        value = e.qr(valuedic['grad_s'], NBIT_DATA_FL)
        self.save_value('grad_s', value, NBIT_DATA)
        # value = e.qr(valuedic['grad_r2_s'], NBIT_DATA_FEA_FL)
        # self.save_value('dy_dr2_s', value, NBIT_DATA_FEA)
        #
        value = e.qr(valuedic['sr'], NBIT_FEA_FL)
        self.save_value('sr', value, NBIT_FEA)
        value = e.qr(valuedic['grad_sr'], NBIT_DATA_FL)
        self.save_value('grad_sr', value, NBIT_DATA)
        # value = e.qr(valuedic['grad_r2_sr'], NBIT_DATA_FEA_FL)
        # self.save_value('dy_dr2_sr', value, NBIT_DATA_FEA)
        #
        value = e.qr(valuedic['G'], NBIT_FEA_FL)
        self.save_value('G', value, NBIT_FEA)
        value = e.qr(valuedic['grad_G'], NBIT_DATA_FL)
        self.save_value('grad_G', value, NBIT_DATA)
        # value = e.qr(valuedic['grad_r2_G'], NBIT_DATA_FEA_FL)
        # self.save_value('dy_dr2_G', value, NBIT_DATA_FEA)
        #
        value = e.qr(valuedic['Rxyz'], NBIT_DATA_FL)
        self.save_value('Rxyz', value, NBIT_DATA)
        value = e.qr(valuedic['grad_Rxyz'], NBIT_DATA_FL)
        self.save_value('grad_Rxyz', value, NBIT_DATA)
        #
        valuedic['GR'] = np.transpose(np.reshape(valuedic['GR'], [-1, 4, M1]), [0, 2, 1])
        value = e.qr(valuedic['GR'], NBIT_DATA_FL).reshape([natom, -1])
        self.save_value('GR', value, NBIT_DATA)
        valuedic['grad_GR'] = np.transpose(np.reshape(valuedic['grad_GR'], [-1, 4, M1]), [0, 2, 1])
        value = e.qf(valuedic['grad_GR'] / NIX, NBIT_DATA_FL).reshape([natom, -1])
        self.save_value('grad_GR', value, NBIT_DATA)
        #
        value = e.qr(valuedic['d'], NBIT_DATA_FL).reshape([natom, -1])
        self.save_value('d', value, NBIT_DATA)
        value = e.qr(valuedic['grad_d'], NBIT_DATA_FL).reshape([natom, -1])
        self.save_value('grad_d', value, NBIT_DATA)

    def save_fitn(self, valuedic):
        natom = self.natom
        nlayer_fit = nvnmd_cfg.fitn['nlayer_fit'] 
        nbit = nvnmd_cfg.nbit 
        NBIT_DATA = nbit['NBIT_DATA']
        NBIT_DATA_FL = nbit['NBIT_DATA_FL']

        e = Encode()
        keys = 'wx,wxb,actfun'.split(',')
        for key in keys:
            for ll in range(nlayer_fit):
                name = f'{key}_l{ll}'
                value = e.qr(valuedic[name], NBIT_DATA_FL)
                self.save_value(name, value, NBIT_DATA)
                name = f'grad_{key}_l{ll}'
                value = e.qr(valuedic[name], NBIT_DATA_FL)
                self.save_value(name, value, NBIT_DATA)
        
    def save_cal_fij(self, valuedic):
        natom = self.natom
        nbit = nvnmd_cfg.nbit 
        NBIT_FORCE = nbit['NBIT_FORCE']
        NBIT_FORCE_FL = nbit['NBIT_FORCE_FL']
        NI = nvnmd_cfg.dscp['NI']

        e = Encode()
        # fij1
        dY_dR2 = valuedic['grad_u'].reshape([natom, -1, 1])
        rij = valuedic['rij']
        fij1 = 2 * dY_dR2 * rij
        value = e.qr(fij1, NBIT_FORCE_FL)
        self.save_value('fij1', value, NBIT_FORCE)
        # fji2
        fij2 = valuedic['grad_rij']
        value = e.qr(fij2, NBIT_FORCE_FL)
        self.save_value('fij2', value, NBIT_FORCE) 
        # fij
        fij = fij1 + fij2 
        value = e.qr(fij, NBIT_FORCE_FL)
        self.save_value('fij', value, NBIT_FORCE) 
        # check the sum in the step3
        # fi and fij_sum
        nlist = np.int32((valuedic['o_nlist']).reshape([natom, -1]))
        fi = np.zeros([natom, 3])
        fij_sum = np.zeros([natom, 3])
        for ii in range(natom):
            for jj in range(NI):
                idx = nlist[ii, jj]
                if idx >= 0:
                    fi[ii]  += fij[ii, jj]
                    fi[idx] -= fij[ii, jj]
                    fij_sum[ii] += fij[ii, jj]
        
        value = e.qr(fi, NBIT_FORCE_FL)
        self.save_value('fi', value, NBIT_FORCE) 

        value = e.qr(fij_sum, NBIT_FORCE_FL)
        self.save_value('fij_sum', value, NBIT_FORCE) 

    def disp_valuedic(self, valuedic):
        for key in valuedic.keys():
            print("%20s"%key, ":", np.shape(valuedic[key]))
        
    def run(self):
        feed_dict = self.get_feed_dic(self.atoms, self.place_holders)
        namedic = self.get_namedic()
        tensordic = self.get_tensordic(namedic)
        valuedic = self.get_valuedic(self.sess, feed_dict, tensordic)
        #
        head = FioHead().info()
        print(f"{head} : valuedic")
        # self.disp_valuedic(valuedic)
        #
        print(f"{head} : s1_merge")
        valuedic = self.s1_merge(valuedic)
        # self.disp_valuedic(valuedic)
        #
        print(f"{head} : s2_resort")
        valuedic = self.s2_resort(valuedic)
        self.disp_valuedic(valuedic)
        #
        self.s3_save(valuedic)

def debug(*,
        nvnmd_config: str = 'nvnmd/config.npy', 
        nvnmd_weight: str = 'nvnmd/weight.npy', 
        nvnmd_map: str = 'nvnmd/map.npy', 
        nvnmd_debug: str = 'nvnmd/debug/', 
        atoms_file: str = 'atoms.xsf', 
        type_map: str = 'type_map.raw', 
        **kwargs
        ):
        debugObj = Debug(
            nvnmd_config, 
            nvnmd_weight, 
            nvnmd_map, 
            nvnmd_debug,
            atoms_file, 
            type_map
            )
        debugObj.build() 
        debugObj.init_model()
        debugObj.run()
