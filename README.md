## StrassenNets

Code to reproduce the experiments in the paper [StrassenNets: Deep learning with a multiplication budget, M. Tschannen, A. Khanna, A. Anandkumar, 2018](https://arxiv.org/abs/1712.03942). In a nutshell, the proposed approach approximates the (generalized) matrix multiplications in deep neural network (DNN) layers by 2-layer sum-product networks (SPNs) and learns the SPNs end-to-end. The CIFAR10 code (Jupyter Notebook) gives an overview of the method.

### Requirements

The code was tested using Python 3.5 and [MXNet](https://github.com/apache/incubator-mxnet) 1.0. The language model code only runs on GPU.

### CIFAR-10

To train Strassen-ResNet-20 on CIFAR-10, use StrassenNetsCIFAR10.ipynb with the [Jupyter Notebook](http://jupyter.org/). For the VGG-inspired 7-layer architecture, run
```
python3 StrassenNetworksCIFAR10_VGG.py --nbr_mul 1 --out_patch 1 --gpu_index 0
```
The multiplication budget can be adjusted using `--nbr_mul` and `--out_patch`.

### Penn Tree Bank (PTB)

1. Download the preprocessed PTB training, testing, and validation files e.g. from [here](https://github.com/yoonkim/lstm-char-cnn/tree/master/data/ptb) (the default location used by the training script is `langmod/data/ptb/`, adapt it using the option `--data_dir` in the command below)
2. Use
```
python3 train.py --quant_mode 5 --max_epochs 40 --decay_when 0.5 --learning_rate 2
```
(in `langmod/`) to train the full precision model, and
```
python3 train.py --quant_mode 2 --max_epochs 20 --decay_when 0.5 --nbr_mul 1.0 --out_nbr_mul 2000 --learning_rate 0.2 --epochs_pre 20 --learning_rate_pre 2
```
to train a Strassen language model. To change the multiplication budget, adapt `--nbr_mul` (budget of all but the last layer, relative to the number of hidden units) and `--out_nbr_mul` (budget of the last layer). The log files and trained models are stored in the folder `langmod/logs/`. To train with a full precision teacher model, use the option `--teacher_model <path to teacher model>`.
