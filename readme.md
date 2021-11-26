

# 文件说明
bin              包含训练，测试，以及生成模型的代码  
dataset       训练数据，包括训练集与测试集，参考文献Huang, J. et al. Deep potential generation scheme and simulation protocol for the Li10GeP2S12-type superionic conductors. J. Chem. Phys. 154, (2021).  
train.json       连续训练的输入文件  
train2.json      量化训练的输入文件



# 前期准备
## 安装依赖
安装DeePMD-kit[https://github.com/deepmodeling/deepmd-kit]

## 编译代码
假设当前目录的绝对路径为 ${ws}
假设 bin 目录的绝对路径为 ${bin} 即 ${ws}/bin

> cd ${bin}/train/tfOp
> bash compile.sh




# 连续网络训练

> cd ${ws}
> mkdir ws-1
> cd ws-1
> mkdir s1
> cd s1
> cp ../../train.json train.json

## 训练网络
> python ${bin}/train/main.py train train.json

## 冻结模型
> python ${bin}/train/main.py freeze -o graph.pb

## 测试精度
> mkdir test-test
> python ${bin}/train/main.py test -m ./graph.pb \
>        -s ../../dataset/lgps-test/ \
>        -d ./test-test/detail -n 999999999 | tee test-test/output

## 冻结参数
> python ${bin}/verilog/main.py freeze weight.npy config.npy

## 生成featureNet映射表
> python ${bin}/verilog/main.py map config.npy map.npy

> cd ../




# 训练量化网络

> mkdir s2
> cd s2
> mkdir old-data
> cp ../s1/config.npy ../s1/model.npy ../s1/map.npy old-data
> cp ../../train2.json train.json

## 训练网络
> python ${bin}/train/main.py train train.json

## 冻结模型
> python ${bin}/train/main.py freeze -o graph.pb

## 测试精度
> mkdir test-test
> python ${bin}/train/main.py test -m ./graph.pb \
>        -s ../../dataset/lgps-test/ \
>        -d ./test-test/detail -n 999999999 | tee test-test/output

## 冻结参数
> python ${bin}/verilog/main.py freeze weight.npy config.npy

## 生成FPGA模型
> python ${bin}/verilog/main.py wrap config.npy old/map.npy fpga_model.pb






