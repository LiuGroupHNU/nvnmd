<span style="font-size:larger;">NVNMD Manual</span>
========

# Table of contents
- [Introduction](#introduction)
- [Preparation](#preparation)
- [Training](#training)
 	- [CNN training](#cnn-training)
 	- [QNN training](#qnn-training)
- [Running MD](#running-md)

# Introduction

NVNMD stands for non-von Neumann molecular dynamics.

Any user can follow two consecutive steps to run molecular dynamics (MD) on the proposed NVNMD computer, which has been released online: (i) to train a machine learning (ML) model that can decently reproduce the potential energy surface (PES); and (ii) to deploy the trained ML model on the proposed NVNMD computer, then run MD there to obtain the atomistic trajectories. You can follow the [user guide](doc/User-guide.pdf) to use NVNMD.

The codes have been migrated into DeePMD-kit, so you can use not only NVNMD but also DeePMD-kit after installation. You can see the [instruction](doc/DeePMD-kit.md) for more details to use DeePMD-kit.

# Preparation

The installation method is the same as that of DeePMD-kit. One may manually download and install NVNMD by following the instructions on [installing from source](doc/install/install-from-source.md). 

# Training

Our training procedure consists of not only the CNN training, but also the QNN training which uses the results of CNN as inputs. It is performed on CPU or GPU by using the training codes we open-sourced online.

To train a ML model that can decently reproduce the PES, training and testing data set should be prepared first. This can be done by using either the state-of-the-art active learning tools, or the outdated (i.e., less efficient) brute-force density functional theory (DFT)-based ab-initio molecular dynamics (AIMD) sampling.

Then, copy the data set to working directory. `$nvnmd_source_dir/examples/nvnmd/data` is the path to the data set used in this example.

## CNN training

### Input script

`$nvnmd_source_dir/examples/nvnmd/train-1.json` is the input script for CNN training. The structure is as follows:

```json
{
    "model": {...},
    "nvnmd": {...},
    "learning_rate": {...},
    "loss": {...},
    "training": {...}
}
```

A model has two parts, a descriptor that maps atomic configuration to a set of symmetry invariant features, and a fitting net that takes descriptor as input and predicts the atomic contribution to the target physical property. It's defined in the `model` section, for example:

```json
    "model": {
        "descriptor": {
            "seed": 1,
            "type": "se_a",
            "sel": [60, 60],
            "rcut": 7.0,
            "rcut_smth": 0.5,
            "neuron": [5, 10, 20],
            "type_one_side": true,
            "axis_neuron": 10,
            "resnet_dt": false
        },
        "fitting_net": {
            "seed": 1,
            "neuron": [20, 20, 20],
            "resnet_dt": false
        }
    },
```

* `model/descriptor/type` should be set to `"se_a"`. 
* `model/descriptor/sel` gives the maximum possible number of neighbors in the cut-off radius. It is a list, the length of which is the same as the number of atom types in the system, and sel[i] denote the maximum possible number of neighbors with type i. 
* `model/descriptor/rcut` is the cut-off radius for neighbor searching.
* `model/descriptor/rcut_smth` gives where the smoothing starts. 
* `model/descriptor/neuron` should be set to `[5, 10, 20]`, which specifies the size of the embedding net. From left to right the members denote the sizes of each hidden layer from input end to the output end, respectively.
* `model/descriptor/type_one_side` should be set to `true` so that descriptor will only consider the types of neighbor atoms. Otherwise, both the types of centric and neighbor atoms are considered.
* `model/descriptor/axis_neuron` should be set to `10`, which specifies the size of submatrix of the embedding matrix, the axis matrix as explained in the DeepPot-SE paper (available at https://arxiv.org/abs/1805.09003). 
* `model/fitting_net/neuron` should be set to `[20, 20, 20]`, which specifies the size of the fitting net.
* `resnet_dt` should be set to `false`, then a timestep is not used in the ResNet.

The `nvnmd` section is defined as follows:

```json
	"nvnmd":{
        "config_file":"none",
        "weight_file":"none",
        "map_file":"none",
        "enable":true,
        "restore_descriptor":false,
        "restore_fitting_net": false,
        "quantize_descriptor":false,
        "quantize_fitting_net":false
    },
```

* `nvnmd/config_file` is used to load the configuration file, which should be set to `"none"` for CNN training.
* `nvnmd/weight_file` is used to load the weight file, which should be set to `"none"` for CNN training.
* `nvnmd/map_file` is used to load the mapping table, which should be set to `"none"` for CNN training.
* `nvnmd/enable` is used to determine whether to use NVNMD, which should be set to `true`.
* `nvnmd/restore_descriptor` and `nvnmd/restore_fitting_net` is used to determine whether to restore the trained model and parameters, which should be set to `false` for CNN training.
* `nvnmd/quantize_descriptor` and `nvnmd/quantize_fitting_net` is used to determine whether to quantize the weights and activations, which should be set to `false` for CNN training.

Goto the working directory and make a training place, then create a directory for CNN training and copy the input script to new directory.

```bash
cd $nvnmd_worspace
mkdir ws-1
cd ws-1
mkdir s1
cd s1
cp $nvnmd_source_dir/examples/nvnmd/train-1.json train.json
```

You can modify the value of parameters in the input script as needed.

### Training

```bash
dp train train.json
```

### Freezing the model

```bash
dp freeze -o graph.pb -w nvnmd/weight.npy
```

### Testing

```bash
mkdir test-test
dp test -m ./graph.pb -s /path/to/system -d ./test-test/detail -n 999999999 | tee test-test/output
```
### Building the mapping table

```bash
dp map -c nvnmd/config.npy -w nvnmd/weight.npy -m nvnmd/map.npy
cd ../
```

## QNN training

### Input script

Compared with the input script for CNN training, some parameters for QNN training need to be modified. An example is provided as follows:

```json
    "nvnmd":{
        "config_file":"../s1/nvnmd/config.npy",
        "weight_file":"../s1/nvnmd/weight.npy",
        "map_file":"../s1/nvnmd/map.npy",
        "enable":true,
        "restore_descriptor":true,
        "restore_fitting_net":true,
        "quantize_descriptor":true,
        "quantize_fitting_net":true
    },

    "learning_rate": {
        "start_lr": 0.0000005,
        ...
    },

    "training": {
        "stop_batch": 10000,
        ...
    }
```

* `nvnmd/config_file` should be set to `"../s1/nvnmd/config.npy"`.
* `nvnmd/weight_file` should be set to `"../s1/nvnmd/weight.npy"`.
* `nvnmd/map_file` should be set to `"../s1/nvnmd/map.npy"`.
* `nvnmd/restore_descriptor` and `nvnmd/restore_fitting_net` should be set to `true` for QNN training to restore parameters obtained by CNN training.
* `nvnmd/quantize_descriptor` and `nvnmd/quantize_fitting_net` should be set to `true` for QNN training to quantize the weights and activations.

Typically, CNN training uses a large number of training steps with a high learning rate; and the subsequent QNN training uses a small number of training steps (e.g., 1×10<sup>4</sup>) and a low learning rate (e.g., 5×10<sup>-7</sup>), as it only needs to minimize the small error induced by quantization from CNN to QNN.

Create a directory for QNN training and copy the input script to new directory

```bash
mkdir s2
cd s2
cp $nvnmd_source_dir/examples/nvnmd/train-2.json train.json
```

You can modify the value of parameters in the input script as needed.

### Training

```bash
dp train train.json
```

### Freezing the model

```bash
dp freeze -o graph.pb -w nvnmd/weight.npy
```

### Testing

```bash
mkdir test-test
dp test -m ./graph.pb -s /path/to/system -d ./test-test/detail -n 999999999 | tee test-test/output
```
### Wrapping the ML model

```bash
dp wrap -c nvnmd/config.npy -w nvnmd/weight.npy -m ../s1/nvnmd/map.npy -o nvnmd/model.pb
```

# Running MD

Upload the QNN model to our online NVNMD system (available at http://nvnmd.picp.vip), then run molecular dynamics simulation there.
