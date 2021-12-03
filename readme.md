# Table of contents
- [File structure](#file-structure)
- [Preparation](#preparation)
- [CNN training](#cnn-training)
- [QNN training](#qnn-training)
- [Inference](#inference)

# File structure

* `bin`: source code of training a machine learning (ML) model  and test.
* `dataset`: data sets including training data set and test data set. See the [paper][1] for more details of the data set in this example.
* `train.json`: input script of  the continuous neural network (CNN) training.
* `train2.json`: input script of  the quantized neural network (QNN) training.

# Preparation

## Installing the dependencies

Download and install [DeePMD-kit][2]. (Currently only version 1.x is supported.)

## Compiling code

`$nvnmd_source_dir` is the absolute path to the current directory.

```bash
$ cd $nvnmd_source_dir/bin/train/tfOp
$ bash compile.sh
```

# CNN training

## Input script

`train.json` is the input script of  CNN training. The structure is as follows

```json
{
    "model": { … },
    "learning_rate": { … },
    "loss": { … },
    "training": { … }
}
```
An example of `model` section is provided as follows
```json
	"model": {
		"descriptor": {
			"type": "se_a",
			"sel": [46, 20, 20, 46],
			"rcut_smth": 0.5,
			"rcut": 6.0,
			"neuron": [8, 16, 32],
			"cf":{
				"quantify":false,
				"trainable":true,
				"retrain":false,
				"NI":128,
				"same_net":true
			},
			"resnet_dt": false,
			"axis_neuron": 4,
			"seed": 936397195
		},
		"fitting_net": {
			"n_neuron": [128, 128, 128],
			"cf":{
				"quantify":false,
				"quantify_grad":false,
				"trainable":true,
				"retrain":false
			},
			"resnet_dt": false,
			"seed": 4074763693
		},
		"type_map": ["Li", "Ge", "P", "S"]
	}
```

* `model/descriptor/type` must be set to `"se_a"`. 
* `model/descriptor/sel` is optional, which gives the maximum possible number of neighbors in the cut-off radius. It is a list, the length of which is the same as the number of atom types in the system, and `sel[i]` denote the maximum possible number of neighbors with type `i`. 
* `model/descriptor/rcut` is optional, which is the cut-off radius for neighbor searching.
* `model/descriptor/rcut_smth` is optional, which gives where the smoothing starts. 
* `model/descriptor/neuron` must be set to `[8, 16, 32]`, which specifies the size of the embedding net. From left to right the members denote the sizes of each hidden layer from input end to the output end, respectively.
* `model/descriptor/cf/quantify` must be set to `false` without quantification.
* `model/descriptor/cf/trainable` must be set to `true` for updating parameters.
* `model/descriptor/cf/retrain` must be set to `false` without reloading parameters.
* `model/descriptor/cf/NI` is exponent of 2 closest to the sum of the maximum possible numbers of neighbors with all atom types. 
* `model/descriptor/cf/same_net` must be set to `true`.
* `model/descriptor/axis_neuron` must be set to `4`, which specifies the size of submatrix of the embedding matrix, the axis matrix as explained in the [DeepPot-SE paper][3]. 
* `model/fitting_net/n_neuron` must be set to `[128, 128, 128]`, which specifies the size of the fitting net. 
* `model/fitting_net/cf/quantify` must be set to `false` without forward quantification.
* `model/fitting_net/cf/quantify_grad` must be set to `false` without backward quantification.
* `model/fitting_net/cf/trainable` must be set to `true` for updating parameters.
* `model/fitting_net/cf/retrain` must be set to `false` without reloading parameters.
* `model/type_map` is optional, which provide the element names for corresponding atom types.
* `resnet_dt` must be set to `false`, then a timestep is not used in the ResNet.
* `seed` gives the random seed that is used to generate random numbers when initializing the model parameters.

## Training

```bash
$ cd $nvnmd_source_dir
$ mkdir ws-1
$ cd ws-1
$ mkdir s1
$ cd s1
$ cp ../../train.json train.json
$ python $nvnmd_source_dir/bin/train/main.py train train.json
```

## Freezing model

```bash
$ python $nvnmd_source_dir/bin/train/main.py freeze -o graph.pb
```

## Test

```bash
$ mkdir test-test
$ python $nvnmd_source_dir/bin/train/main.py test -m ./graph.pb -s ../../dataset/lgps-test/ -d ./test-test/detail -n 999999999 | tee test-test/output
```

## Freezing parameters

```bash
$ python $nvnmd_source_dir/bin/verilog/main.py freeze weight.npy config.npy
```

## Generating featureNet mapping table

```bash
$ python $nvnmd_source_dir/bin/verilog/main.py map config.npy map.npy
$ cd ../
```

# QNN training

The whole training procedure consists of not only the CNN training, but also the QNN training which uses CNN results as inputs.

## Input script

`train2.json` is the input script of QNN training. Compared with the input script of CNN training, some parameters need to be modified. An example is provided as follows

```json
	"model/descriptor/cf":{
 				"quantify":true,
 				"trainable":false,
 				"retrain":true,
 				"NI":128,
 				"same_net":true
	}

	"model/fitting_net/cf":{
				"quantify":true,
				"quantify_grad":true,
				"trainable":true,
				"retrain":true
	}

	"learning_rate": {
		"type": "exp",
		"start_lr": 0.0000002,
		"decay_steps": 2000,
		"decay_rate": 0.95
	}

	"training": {
		"stop_batch": 20000,
		…
	}
```

* `model/descriptor/cf/quantify` must be set to `true` for quantification.
* `model/descriptor/cf/trainable` must be set to `false` without updating parameters.
* `model/descriptor/cf/retrain` must be set to `true` for reloading parameters.
* `model/fitting_net/cf/quantify` must be set to `true` for forward quantification.
* `model/fitting_net/cf/quantify_grad` must be set to `true` for backward quantification.
* `model/fitting_net/cf/retrain` must be set to `true` for reloading parameters.
* `learning_rate/start_lr` should be lower (e.g., 2×e-7) and `training/stop_batch` should be smaller (e.g., 2×e4), as it only needs to minimize the small error induced by quantization from CNN to QNN.

## Training

```bash
$ mkdir s2
$ cd s2
$ mkdir old-data
$ cp ../s1/config.npy ../s1/model.npy ../s1/map.npy old-data
$ cp ../../train2.json train.json
$ python $nvnmd_source_dir/bin/train/main.py train train.json
```

## Freezing model

```bash
$ python $nvnmd_source_dir/bin/train/main.py freeze -o graph.pb
```

## Test

```bash
$ mkdir test-test
$ python $nvnmd_source_dir/bin/train/main.py test -m ./graph.pb -s ../../dataset/lgps-test/ -d ./test-test/detail -n 999999999 | tee test-test/output
```

## Freezing parameters

```bash
$ python $nvnmd_source_dir/bin/verilog/main.py freeze weight.npy config.npy
```

## Generating FPGA model

```bash
$ python $nvnmd_source_dir/bin/verilog/main.py wrap config.npy old-data/map.npy fpga_model.pb
```

# Inference

## Required files

* `input script`: pair_style command must be set to `"pair_style fpga Rc"`, where Rc is global cutoff. An example is provided as follows

```
  pair_style fpga 6.0
```
* `parameter file(fpga_model.pb)`: FPGA model generated by QNN training.
* `data files`: data files containing information LAMMPS needs to run a simulation.

## Running 

Upload the QNN model to our [online NVNMD system][4], then run molecular dynamics simulation there.

[1]: https://aip.scitation.org/doi/10.1063/5.0041849
[2]: https://github.com/deepmodeling/deepmd-kit
[3]: https://arxiv.org/abs/1805.09003
[4]: http://nvnmd.picp.vip
