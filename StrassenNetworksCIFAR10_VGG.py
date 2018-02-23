import mxnet as mx
import numpy as np
import argparse
from mxnet import gluon, autograd
from mxnet.base import numeric_types
from mxnet.gluon import nn
from math import ceil

from common.sumprodlayer import *

parser = argparse.ArgumentParser(description='SP-ResNet-18 on CIFAR-10.')
parser.add_argument('--nbr_mul', type=float, default=1.0,
                    help='nbr_mul')
parser.add_argument('--out_patch', type=int, default=1,
                    help='out_patch')
parser.add_argument('--gpu_idx', type=int, default=0,
                    help='index of gpu')

opt = parser.parse_args()

class VGG7SumProd(gluon.HybridBlock):

    def __init__(self, channels, nbr_mul, out_patch, classes=10, prefix='vgg7_', **kwargs):
        super(VGG7SumProd, self).__init__(prefix, **kwargs)

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')

            prev_ch = 3
            for i in range(len(channels)-1):
                tshape1 = (channels[i], prev_ch, 3, 3)
                self.features.add(SumProd2DConv(nbr_mul = nbr_mul[i],
                                              target_layer_shape = tshape1,
                                              target_layer_key = 'conv1',
                                              kernel = (3, 3),
                                              stride = (1, 1),
                                              out_patch = out_patch[i],
                                              prefix = 'conv%i-1_'%i))
                self.features.add(nn.BatchNorm())
                self.features.add(nn.Activation('relu'))

                tshape2 = (channels[i], channels[i], 3, 3)
                self.features.add(SumProd2DConv(nbr_mul = nbr_mul[i],
                                              target_layer_shape = tshape2,
                                              target_layer_key = 'conv1',
                                              kernel = (3, 3),
                                              stride = (1, 1),
                                              out_patch = out_patch[i],
                                              prefix = 'conv%i-2_'%i))
                self.features.add(nn.MaxPool2D())
                self.features.add(nn.BatchNorm())
                self.features.add(nn.Activation('relu'))
                prev_ch = channels[i]

            self.classifier = nn.HybridSequential(prefix='')
            self.classifier.add(nn.Flatten())
            self.classifier.add(SumProdDense(nbr_mul=1024,
                                            units=1024,
                                            in_units=channels[-1],
                                            use_bias=True,
                                            use_batchnorm=True))
            self.classifier.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.classifier.add(nn.Dense(classes, in_units=1024))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


classes = 10
channels = [128, 256, 512, 8192]
print('nbr_mul: %f'%opt.nbr_mul)
nbr_mul = [int(i*opt.nbr_mul) for i in channels]
print('out_patch: %f'%opt.out_patch)
out_patch = [(opt.out_patch, opt.out_patch)]*3
sp_vgg = VGG7SumProd(channels, nbr_mul, out_patch, classes)

print('gpu_idx: %f'%opt.gpu_idx)
ctx = [mx.gpu(opt.gpu_idx)]

# helper function to filter and intialize paramters whose name contains a given substring
def initialize_by_key(net, key, init, context, force_reinit=False):
    for k, p in net.collect_params().items():
        if key in k:
            p.initialize(init=init, ctx=context, force_reinit=force_reinit)

# initialize SP networks
initialize_by_key(sp_vgg, 'sp_mode', mx.init.One(), ctx)
initialize_by_key(sp_vgg, 'sp_in_weights', mx.init.One(), ctx)
initialize_by_key(sp_vgg, 'sp_data_weights', mx.init.Xavier(magnitude=2), ctx)
initialize_by_key(sp_vgg, 'sp_out_weights', mx.init.Xavier(magnitude=2), ctx)

# intialize batchnorm parameters
initialize_by_key(sp_vgg, 'gamma', mx.init.One(), ctx)
initialize_by_key(sp_vgg, 'beta', mx.init.Zero(), ctx)
initialize_by_key(sp_vgg, 'running_mean', mx.init.Zero(), ctx)
initialize_by_key(sp_vgg, 'running_var', mx.init.Zero(), ctx)

# initialize fully connected layer
initialize_by_key(sp_vgg, 'dense0_weight', mx.init.Xavier(magnitude=2), ctx) #mx.init.Normal(1)
initialize_by_key(sp_vgg, 'dense0_bias', mx.init.Zero(), ctx)

# deactivate optimization for original convolutions as they are not used here
for k, p in sp_vgg.collect_params().items():
    if 'conv' in k and 'sp' not in k:
        p.initialize(init=mx.init.Xavier(magnitude=2), ctx=ctx)
        p.grad_req = 'null'

# deactivate optimization for batchnorm auxiliary variables
for k, p in sp_vgg.collect_params().items():
    if 'running_mean' in k or 'running_var' in k or 'sp_mode' in k:
        p.grad_req = 'null'

sp_vgg.collect_params()['vgg7_sumproddense0_weight'].grad_req = 'null'
sp_vgg.collect_params()['vgg7_sumproddense0_bias'].grad_req = 'null'

import time

def test(net, val_data, context):
    metric = mx.metric.Accuracy()
    val_data.reset()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=context, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=context, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x))
        metric.update(label, outputs)
    return metric.get()


def train(net, lr, mom, wd, epochs, scheduler, context, train_data, val_data, log_interval):
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'wd': wd, 'momentum': mom, 'lr_scheduler': scheduler},
                            kvstore = 'device')
    metric = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    for epoch in range(epochs):
        tic = time.time()
        train_data.reset()
        metric.reset()
        btic = time.time()
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            Ls = []
            with autograd.record():
                for x, y in zip(data, label):
                    z = net(x)
                    L = loss(z, y)
                    Ls.append(L)
                    outputs.append(z)
                for L in Ls:
                    L.backward()
            trainer.step(batch.data[0].shape[0])
            metric.update(label, outputs)
            if log_interval and not (i+1)%log_interval:
                name, acc = metric.get()
                print('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f'%(
                               epoch, i, batch_size/(time.time()-btic), name, acc))
            btic = time.time()

        name, acc = metric.get()
        print('[Epoch %d] training: %s=%f'%(epoch, name, acc))
        print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
        name, val_acc = test(net, val_data, context)
        print('[Epoch %d] validation: %s=%f'%(epoch, name, val_acc))

mx.random.seed(321)
np.random.seed(321)

data_shape = (3, 32, 32)
batch_size = 128

# download CIFAR-10 if necessary
import os.path
if (not os.path.isfile("data/cifar/train.rec")) or (not os.path.isfile("data/cifar/test.rec")):
    zip_file_path = mx.test_utils.download('http://data.mxnet.io/mxnet/data/cifar10.zip', dirname='data')
    import zipfile
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall('data')

# training set iterator
train_data = mx.io.ImageRecordIter(
    path_imgrec   = "data/cifar/train.rec",
    data_shape    = data_shape,
    batch_size    = batch_size,
    mean_r        = 125.3,
    mean_g        = 123.0,
    mean_b        = 113.9,
    std_r         = 63.0,
    std_g         = 62.1,
    std_b         = 66.7,
    dtype         = 'float32',
    rand_crop     = True,
    max_crop_size = 32,
    min_crop_size = 32,
    pad           = 4,
    fill_value    = 0,
    shuffle       = True,
    rand_mirror   = True,
    shuffle_chunk_seed  = 123)

# validation set iterator
val_data = mx.io.ImageRecordIter(
    mean_r      = 125.3,
    mean_g      = 123.0,
    mean_b      = 113.9,
    std_r       = 63.0,
    std_g       = 62.1,
    std_b       = 66.7,
    dtype       = 'float32',
    path_imgrec = "data/cifar/test.rec",
    rand_crop   = False,
    rand_mirror = False,
    data_shape  = data_shape,
    batch_size  = batch_size)

epochs = 250
learning_rate = 0.1
momentum = 0.9
weight_decay = 0.0001

train_set_size = 50000
schedule_factor = ceil(train_set_size/batch_size)
scheduler = mx.lr_scheduler.MultiFactorScheduler([150*schedule_factor, 200*schedule_factor, 250*schedule_factor], factor=0.1)

log_interval = 50

train(sp_vgg, learning_rate, momentum, weight_decay, epochs, scheduler, ctx, train_data, val_data, log_interval)

initialize_by_key(sp_vgg, 'sp_mode', mx.init.Constant(2), ctx, True)

epochs = 40
learning_rate = 0.01
scheduler = mx.lr_scheduler.MultiFactorScheduler([10*schedule_factor, 20*schedule_factor, 30*schedule_factor, 40*schedule_factor], factor=0.1)

train(sp_vgg, learning_rate, momentum, weight_decay, epochs, scheduler, ctx, train_data, val_data, log_interval)


for k, p in sp_vgg.collect_params().items():
    if 'sp_data_weights' in k or 'sp_out_weights' in k:
        p.grad_req = 'null'

epochs = 10
learning_rate = 0.001
scheduler = None

train(sp_vgg, learning_rate, momentum, weight_decay, epochs, scheduler, ctx, train_data, val_data, log_interval)

sp_vgg.collect_params().save('sp_vgg.params')
