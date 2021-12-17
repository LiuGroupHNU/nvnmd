
import numpy as np

from deepmd.nvnmd.data.data import jdata_config, jdata_deepmd_input
from deepmd.nvnmd.utils.fio import FioDic, FioHead

class NvnmdConfig():
    def __init__(self,
        jdata: dict
        ) -> None:
        self.map = {} 
        self.config = jdata_config
        self.weight = {}
        self.init_from_jdata(jdata)

    def init_from_jdata(self, jdata: dict={}):
        head = FioHead().info()
        print(f"{head}: configure the nvnmd")

        if jdata == {}:
            return None

        self.map_file = jdata['map_file']
        self.config_file = jdata['config_file']
        self.enable = jdata['enable']
        self.weight_file = jdata['weight_file']
        self.restore_descriptor = jdata['restore_descriptor']
        self.restore_fitting_net = jdata['restore_fitting_net']
        self.quantize_descriptor = jdata['quantize_descriptor']
        self.quantize_fitting_net = jdata['quantize_fitting_net']

        # load data
        if self.enable:
            self.map = FioDic().load(self.map_file, {})
            self.weight = FioDic().load(self.weight_file, {})
            load_config = FioDic().load(self.config_file, jdata_config)
            self.init_from_config(load_config)
        
    
    def init_update(self, jdata, jdata_o):
        """
        jdata: new jdata
        jdata_o: origin jdata
        """
        for key in jdata.keys():
            if key in jdata_o.keys():
                if type(jdata_o[key]) == dict:
                    jdata_o[key] = self.init_update(jdata[key], jdata_o[key])
                else:
                    jdata_o[key] = jdata[key]
        return jdata_o

    def init_value(self):
        self.dscp = self.config['dscp']
        self.fitn = self.config['fitn']
        self.size = self.config['size']
        self.ctrl = self.config['ctrl']
        self.nbit = self.config['nbit'] 

    def init_from_config(self, jdata):
        self.config = self.init_update(jdata, self.config)
        self.config['dscp'] = self.init_dscp(self.config['dscp'], self.config)
        self.config['fitn'] = self.init_fitn(self.config['fitn'], self.config)
        self.config['size'] = self.init_size(self.config['size'], self.config)
        self.config['ctrl'] = self.init_ctrl(self.config['ctrl'], self.config)
        self.config['nbit'] = self.init_nbit(self.config['nbit'], self.config)
        self.init_value()

    def init_from_deepmd_input(self, jdata):
        self.config['dscp'] = self.init_update(jdata['descriptor'], self.config['dscp'])
        self.config['fitn'] = self.init_update(jdata['fitting_net'], self.config['fitn'])
        self.config['dscp'] = self.init_dscp(self.config['dscp'], self.config)
        self.config['fitn'] = self.init_fitn(self.config['fitn'], self.config)
        self.init_value()

    def init_dscp(self, jdata: dict, jdata_parent: dict={}) -> dict:
        jdata['M1'] = jdata['neuron'][-1]
        jdata['M2'] = jdata['axis_neuron']
        jdata['NNODE_FEAS'] = [1] + jdata['neuron']
        jdata['nlayer_fea'] = len(jdata['neuron'])
        jdata['same_net'] = jdata['type_one_side']
        jdata['NIDP'] = int(np.sum(jdata['sel'])) 
        jdata['NIX'] = 2 ** int(np.ceil(np.log2(jdata['NIDP'] / 1.5)))
        jdata['SEL'] = (jdata['sel'] + [0,0,0,0])[0:4]
        jdata['ntype'] = len(jdata['sel'])
        jdata['ntypex'] = 1 if(jdata['same_net']) else jdata['ntype']
        
        return jdata 

    def init_fitn(self, jdata: dict, jdata_parent: dict={}) -> dict:
        M1 = jdata_parent['dscp']['M1']
        M2 = jdata_parent['dscp']['M2']

        jdata['NNODE_FITS'] = [int(M1*M2)] + jdata['neuron'] + [1]
        jdata['nlayer_fit'] = len(jdata['neuron']) + 1
        jdata['NLAYER'] = jdata['nlayer_fit']

        return jdata 

    def init_size(self, jdata: dict, jdata_parent: dict={}) -> dict:
        jdata['Na'] = jdata['NSPU']
        jdata['NaX'] = jdata['MSPU']
        return jdata    
    
    def init_ctrl(self, jdata: dict, jdata_parent: dict={}) -> dict:
        ntype_max = jdata_parent['dscp']['ntype_max']
        jdata['NSADV'] = jdata['NSTDM'] + 1
        jdata['NSEL'] = jdata['NSTDM'] * ntype_max 
        return jdata 
    
    def init_nbit(self, jdata: dict, jdata_parent: dict={}) -> dict:
        Na = jdata_parent['size']['Na']
        NaX = jdata_parent['size']['NaX']
        jdata['NBIT_CRD'] = jdata['NBIT_DATA'] * 3
        jdata['NBIT_LST'] = int(np.ceil(np.log2(NaX)))
        jdata['NBIT_ATOM'] = jdata['NBIT_SPE'] + jdata['NBIT_CRD']
        jdata['NBIT_LONG_ATOM'] = jdata['NBIT_SPE'] + jdata['NBIT_LONG_DATA'] * 3
        jdata['NBIT_RIJ'] = jdata['NBIT_DATA_FL'] + 5
        jdata['NBIT_SUM'] = jdata['NBIT_DATA_FL'] + 8

        return jdata 
    
    def save(self, file_name='nvnmd/config.npy'):
        FioDic().save(file_name, self.config)
    
    def get_dscp_jdata(self):
        dscp = self.dscp
        jdata = jdata_deepmd_input['model']['descriptor']
        jdata['sel'] = dscp['sel']
        jdata['rcut'] = dscp['rcut']
        jdata['rcut_smth'] = dscp['rcut_smth']
        jdata['neuron'] = dscp['neuron']
        jdata['type_one_side'] = dscp['type_one_side']
        jdata['axis_neuron'] = dscp['axis_neuron']
        return jdata 
    
    def get_fitn_jdata(self):
        fitn = self.fitn 
        jdata = jdata_deepmd_input['model']['fitting_net']
        jdata['neuron'] = fitn['neuron']
        return jdata 
    
    def get_model_jdata(self):
        jdata = jdata_deepmd_input['model']
        jdata['descriptor'] = self.get_dscp_jdata()
        jdata['fitting_net'] = self.get_fitn_jdata()
        return jdata 
    
    def get_nvnmd_jdata(self):
        jdata = jdata_deepmd_input['nvnmd']
        jdata['config_file'] = self.config_file
        jdata['weight_file'] = self.weight_file
        jdata['map_file'] = self.map_file
        jdata['enable'] = self.enable
        jdata['restore_descriptor'] = self.restore_descriptor
        jdata['restore_fitting_net'] = self.restore_fitting_net
        jdata['quantize_descriptor'] = self.quantize_descriptor
        jdata['quantize_fitting_net'] = self.quantize_fitting_net
        return jdata 
    
    def get_learning_rate_jdata(self):
        return jdata_deepmd_input['learning_rate']
    
    def get_loss_jdata(self):
        return jdata_deepmd_input['loss']
    
    def get_training_jdata(self):
        return jdata_deepmd_input['training']

    def get_deepmd_jdata(self):
        jdata = jdata_deepmd_input.copy()
        jdata['model'] = self.get_model_jdata()
        jdata['nvnmd'] = self.get_nvnmd_jdata()
        jdata['learning_rate'] = self.get_learning_rate_jdata()
        jdata['loss'] = self.get_loss_jdata()
        jdata['training'] = self.get_training_jdata()
        return jdata 

# global configuration for nvnmd
nvnmd_cfg = NvnmdConfig(jdata_deepmd_input['nvnmd'])


