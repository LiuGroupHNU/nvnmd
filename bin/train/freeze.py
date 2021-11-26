#!/usr/bin/env python3

# freeze.py :
# see https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc

from env import tf
from env import op_module

# load grad of force module
import deepmd._prod_force_grad
import deepmd._prod_virial_grad
import deepmd._prod_force_se_a_grad
import deepmd._prod_virial_se_a_grad
import deepmd._prod_force_se_r_grad
import deepmd._prod_virial_se_r_grad
import deepmd._soft_min_force_grad
import deepmd._soft_min_virial_grad

def _make_node_names(model_type, modifier_type = None) :
    if model_type == 'ener':
        nodes = "o_energy,o_force,o_virial,o_atom_energy,o_atom_virial,descrpt_attr/rcut,descrpt_attr/ntypes,fitting_attr/dfparam,fitting_attr/daparam,model_attr/tmap,model_attr/model_type"
    elif model_type == 'wfc':
        nodes = "o_wfc,descrpt_attr/rcut,descrpt_attr/ntypes,model_attr/tmap,model_attr/sel_type,model_attr/model_type"
    elif model_type == 'dipole':
        # nodes = "o_dipole,o_rmat,o_rmat_deriv,o_nlist,o_rij,descrpt_attr/rcut,descrpt_attr/ntypes,descrpt_attr/sel,descrpt_attr/ndescrpt,model_attr/tmap,model_attr/sel_type,model_attr/model_type,model_attr/output_dim"
        nodes = "o_dipole,o_rmat,o_rmat_deriv,o_nlist,o_rij,descrpt_attr/rcut,descrpt_attr/ntypes,descrpt_attr/sel,descrpt_attr/ndescrpt,model_attr/tmap,model_attr/sel_type,model_attr/model_type,model_attr/output_dim,isTraining,o_G,o_F,o_GR,o_descriptor"
        # nodes = "o_dipole,o_rmat,o_rmat_deriv,o_nlist,o_rij,descrpt_attr/rcut,descrpt_attr/ntypes,descrpt_attr/sel,descrpt_attr/ndescrpt,model_attr/tmap,model_attr/sel_type,model_attr/model_type,model_attr/output_dim,isTraining"
    elif model_type == 'polar':
        nodes = "o_polar,descrpt_attr/rcut,descrpt_attr/ntypes,model_attr/tmap,model_attr/sel_type,model_attr/model_type"
    elif model_type == 'global_polar':
        nodes = "o_global_polar,descrpt_attr/rcut,descrpt_attr/ntypes,model_attr/tmap,model_attr/sel_type,model_attr/model_type"
    else:
        raise RuntimeError('unknow model type ' + model_type)
    if modifier_type == 'dipole_charge':
        nodes += ",modifier_attr/type,modifier_attr/mdl_name,modifier_attr/mdl_charge_map,modifier_attr/sys_charge_map,modifier_attr/ewald_h,modifier_attr/ewald_beta,dipole_charge/descrpt_attr/rcut,dipole_charge/descrpt_attr/ntypes,dipole_charge/model_attr/tmap,dipole_charge/model_attr/model_type,o_dm_force,dipole_charge/model_attr/sel_type,dipole_charge/o_dipole,dipole_charge/model_attr/output_dim,o_dm_virial,o_dm_av"
    return nodes

def freeze_graph(model_folder, 
                 output, 
                 output_node_names = None):
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    # We precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/" + output

    # Before exporting our graph, we need to precise what is our output node
    # This is how TF decides what part of the Graph he has to keep and what part it can dump
    # NOTE: this variable is plural, because you can have multiple output nodes
    # output_node_names = "energy_test,force_test,virial_test,t_rcut"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True
    
    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    nodes = [n.name for n in input_graph_def.node]

    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        model_type = sess.run('model_attr/model_type:0', feed_dict = {}).decode('utf-8')
        if 'modifier_attr/type' in nodes:
            modifier_type = sess.run('modifier_attr/type:0', feed_dict = {}).decode('utf-8')
        else:
            modifier_type = None
        if output_node_names is None :
            output_node_names = _make_node_names(model_type, modifier_type)
        print('The following nodes will be frozen: %s' % output_node_names)

        save_weight(sess)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            input_graph_def, # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        ) 

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


# 保存权值
# ---------------------------------------------------------------------
"""

# 功能
save_weight 保存权值
read_input 读取输入文件
change_name 权值的名字更改

"""
# ---------------------------------------------------------------------

import scipy.io as sio
import numpy as np
import json

def change_name(key,cf):
    """
    layer_0_type_0/matrix:0
    final_layer_type_0/matrix:0

    descrpt_attr_t_avg
    descrpt_attr_t_std
    filter_type_0_matrix_1_0
    filter_type_0_bias_1_0
    filter_type_0_matrix_4_0
    filter_type_0_bias_4_0
    layer_0_type_0_matrix
    layer_0_type_0_bias
    final_layer_type_0_matrix
    final_layer_type_0_bias
    """
    pars = key.split('_')
    if 'type' in key:
        if ('filter_' in key):
            key = "fea_t%s_t%s_l%d_%s"%(pars[2], pars[5], int(pars[4])-1, 'm' if ('mat' in key) else 'b')
        else:
            if ('final' in key):
                key = "fit_t%s_l%d_%s"%(pars[3], cf['ox123121110134114127108115118129']-1, 'm' if ('mat' in key) else 'b')
            else:
                key = "fit_t%s_l%s_%s"%(pars[3], pars[1], 'm' if ('mat' in key) else 'b')
    else:
        key = "%s"%(pars[3])
    return key

def read_input(json_fn):
    fr = open(json_fn, 'r')
    jdata = json.load(fr)
    fr.close()

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

    cf = {}
    cf['ox128134122111124121'] = get_value(jdata, 'model.type_map')
    cf['ox123129134125114112118123114'] = len(cf['ox128134122111124121'])
    cf['ox127112128116101'] = get_value(jdata, 'model.descriptor.rcut_smth')
    cf['ox127112'] = get_value(jdata, 'model.descriptor.rcut')
    cf['ox091091092081082108083082078096'] = get_value(jdata, 'model.descriptor.neuron')
    cf['ox090062'] = cf['ox091091092081082108083082078096'][-1]
    cf['ox090063'] = get_value(jdata, 'model.descriptor.axis_neuron')
    cf['ox091091092081082108083086097096'] = get_value(jdata, ['model.fitting_net.neuron', 'model.fitting_net.n_neuron'])
    cf['ox123121110134114127108115114110'] = len(cf['ox091091092081082108083082078096'])
    cf['ox123121110134114127108115118129'] = len(cf['ox091091092081082108083086097096'])+1
    cf['ox128114121117103'] =  get_value(jdata, 'model.descriptor.sel')
    cf['ox091086081093'] =  int(np.sum(cf['ox128114121117103']))
    cf['ox091086101080075'] = get_value(jdata, 'model.descriptor.cf.NI')
    cf['ox128110122114108123114129'] = get_value(jdata, 'model.descriptor.cf.same_net')
    cf['ox123129134125114133'] = 1 if cf['ox128110122114108123114129'] else cf['ox123129134125114112118123114']
    # print(cf)
    return cf

def save_weight(sess):
    tvs = tf.global_variables()
    nameList = [v.name for v in tvs]
    nameList = [name.replace(':0','') for name in nameList]
    nameList = [name.replace('/','_') for name in nameList]
    valueList = [sess.run(v) for v in tvs]
    ws = dict(zip(nameList, valueList))
    #filter
    cf = read_input('train.json')
    ws2 = {}
    ws3 = {}
    for key in ws.keys():
        if ('XXX' not in key) and ('Adam' not in key) and (('type' in key) or 'descrpt' in key):
            key2 = change_name(key, cf)
            ws2[key2] = ws[key]
            ws3[key] = ws[key]
    ws2.update(cf)
    np.save('weight.npy', [ws2], allow_pickle=True)
    np.save('model.npy', [ws3], allow_pickle=True)

def freeze (args):
    freeze_graph(args.folder, args.output, args.nodes)

