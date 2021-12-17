

jdata_config ={
    "dscp" : {
        "sel" : [10, 10],
        "rcut" : 6.0,
        "rcut_smth" : 0.5,
        "neuron": [5, 10, 20],
        "resnet_dt": False,
        "axis_neuron": 10,
        "type_one_side": True,

        "NI": 128,
        "rc_lim": 0.5,
        "M1": "neuron[-1]",
        "M2": "axis_neuron",
        "SEL": [10, 10, 0, 0],
        "NNODE_FEAS": "(1, neuron)",
        "nlayer_fea": "len(neuron)",
        "same_net": "type_one_side",
        "NIDP": "sum(sel)", 
        "NIX": "2^ceil(ln2(NIDP/1.5))",
        "ntype": "len(sel)",
        "ntypex": "same_net ? 1 : ntype",
        "ntypex_max": 1,
        "ntype_max": 4
    },

    "fitn": {
        "neuron": [20, 20, 20],
        "resnet_dt": False,

        "NNODE_FITS": "(M1*M2, neuron, 1)",
        "nlayer_fit": "len(neuron)+1",
        "NLAYER": "nlayer_fit"
    },

    "size" : {
        "NTYPE_MAX": 4,
        "NSPU": 4096,
        "MSPU": 16384,
        "Na": "NSPU",
        "NaX": "MSPU"
    },

    "ctrl" : {
        "NSTDM": 10, 
        "NSTDM_M1": 10, 
        "NSTDM_M2": 1, 
        "NSADV": "NSTDM+1",
        "NSEL" : "NSTDM*ntype_max",  
        "NSTDM_M1X": 1, 
        "NSTEP_DELAY": 20,
        "MAX_FANOUT": 30
    },

    "nbit" : {
        "NBIT_DATA": 21,
        "NBIT_DATA_FL": 13,
        "NBIT_LONG_DATA": 56,
        "NBIT_LONG_DATA_FL": 48,
        "NBIT_DIFF_DATA": 48,
        
        "NBIT_SPE": 2, 
        "NBIT_CRD": "NBIT_DATA*3",
        "NBIT_LST": "ln2(NaX)", 
        
        "NBIT_SPE_MAX": 8, 
        "NBIT_LST_MAX": 16, 
        
        "NBIT_ATOM": "NBIT_SPE+NBIT_CRD", 
        "NBIT_LONG_ATOM": "NBIT_SPE+NBIT_LONG_DATA*3", 
        
        "NBIT_RIJ": "NBIT_DATA_FL+5", 
        "NBIT_FEA_X": 10, 
        "NBIT_FEA": 18, 
        "NBIT_FEA_FL": 10, 
        "NBIT_SHIFT": 4,
        
        "NBIT_SUM": "NBIT_DATA_FL+8", 
        "NBIT_WLN2": 4, 
        
        "NBIT_RAM": 72, 
        "NBIT_ADDR": 32, 
        
        "NBIT_TH_LONG_ADD": 30, 
        "NBIT_ADD": 15,

        "NUM_WLN2" : 3, 
        "RANGE_B" : [-100, 100], 
        "RANGE_W" : [-20, 20]
    },

    "end": ""    
}


jdata_deepmd_input = {
    "model": {
        "descriptor": {
            "seed": 1,
            "type": "se_a",
            "sel": [
                60,
                60
            ],
            "rcut": 7.0,
            "rcut_smth": 0.5,
            "neuron": [
                5,
                10,
                20
            ],
            "type_one_side": False,
            "axis_neuron": 10,
            "resnet_dt": False
        },
        "fitting_net": {
            "seed": 1,
            "neuron": [
                20,
                20,
                20
            ],
            "resnet_dt": False
        }
    },
    "nvnmd":{
        "config_file":"none",
        "weight_file":"none",
        "map_file":"none",
        "enable":False,
        "restore_descriptor":False,
        "restore_fitting_net": False,
        "quantize_descriptor":False,
        "quantize_fitting_net":False
    },
    "learning_rate": {
        "type": "exp",
        "decay_steps": 5000,
        "start_lr": 0.005,
        "stop_lr": 8.257687192506788e-05
    },
    "loss": {
        "start_pref_e": 0.02,
        "limit_pref_e": 1,
        "start_pref_f": 1000,
        "limit_pref_f": 1,
        "start_pref_v": 0,
        "limit_pref_v": 0
    },
    "training": {
        "seed": 1,
        "stop_batch": 10000,
        "disp_file": "lcurve.out",
        "disp_freq": 100,
        "numb_test": 10,
        "save_freq": 1000,
        "save_ckpt": "model.ckpt",
        "disp_training": True,
        "time_training": True,
        "profiling": False,
        "training_data": {
            "systems": "dataset",
            "set_prefix": "set",
            "batch_size": 1
        }
    }
}
