import numpy as np
import mxnet as mx
from mxnet import gluon

from .ter_quant import *

from mxnet.base import numeric_types

from math import ceil

import ast




class SumProd2DConv(mx.gluon.HybridBlock):

    def __init__(self, nbr_mul, target_layer_shape, target_layer_key, out_patch=(1,1), kernel=(3,3), stride=(1,1), pad=(1,1), num_group=1, no_bias=True, layout='NCHW', scale_quant=1.0, clip_weights=False, rescale_quant=False, prefix=None, **kwargs):
        # Use same prefix as parent layer for loading the shared paramters to be compressed
        super(SumProd2DConv, self).__init__(prefix=prefix, **kwargs)

        if isinstance(stride, numeric_types):
            stride = (stride,)*len(kernel)
        if isinstance(pad, numeric_types):
            pad = (pad,)*len(kernel)

        self._nbr_mul = nbr_mul
        self._out_patch = out_patch
        self._kernel = kernel
        self._stride = stride
        self._pad = pad
        self._num_group = num_group
        self._layout = layout
        self._scale_quant = scale_quant
        self._clip_weights = clip_weights
        self._quant_op = 'ter_quant_rescaled' if rescale_quant else 'ter_quant'

        self._sp_data_weight_kernel = ((self._out_patch[0]-1)*self._stride[0]+self._kernel[0], (self._out_patch[1]-1)*self._stride[1]+self._kernel[1])
        self._sp_data_weight_stride = (self._out_patch[0]*self._stride[0], self._out_patch[1]*self._stride[1])

        self.filter_weights = self.params.get(target_layer_key, shape=target_layer_shape)

        filter_weights_shape = list(target_layer_shape)
        sp_data_weights_shape = filter_weights_shape.copy()
        sp_data_weights_shape[0] = nbr_mul
        sp_data_weights_shape[1] //= num_group
        sp_data_weights_shape[2] = self._sp_data_weight_kernel[0]
        sp_data_weights_shape[3] = self._sp_data_weight_kernel[1]
        sp_out_weights_shape = filter_weights_shape.copy()
        sp_out_weights_shape[0] = nbr_mul
        sp_out_weights_shape[1] = filter_weights_shape[0]
        sp_out_weights_shape[2] = self._out_patch[0]
        sp_out_weights_shape[3] = self._out_patch[1]

        self.sp_in_weights = self.params.get('sp_in_weights', shape=(1,self._nbr_mul,1,1))
        self.sp_data_weights = self.params.get('sp_data_weights', shape=sp_data_weights_shape)
        self.sp_out_weights =  self.params.get('sp_out_weights', shape=sp_out_weights_shape)

        self.bias = None if no_bias else self.params.get('bias', shape=(filter_weights_shape[0],), init=mx.init.Zero())

        ## batchnorm
        self.sp_batchnorm_gamma = self.params.get('sp_batchnorm_gamma', shape=(1,self._nbr_mul,1,1))
        self.sp_batchnorm_beta = self.params.get('sp_batchnorm_beta', shape=(1,self._nbr_mul,1,1))
        self.sp_batchnorm_running_mean = self.params.get('sp_batchnorm_running_mean', shape=(1,self._nbr_mul,1,1))
        self.sp_batchnorm_running_var = self.params.get('sp_batchnorm_running_var', shape=(1,self._nbr_mul,1,1))

        '''
        Train mode:
            0: Do full precision operation
            1: Compressed operation, compute forward pass without quantization
            2: Compressed operation, compute forward pass without quantization
        '''
        self.sp_mode = self.params.get('sp_mode', shape=(1,), dtype=np.uint8)


    def forward(self, data):
        with data.context:
            mode = self.sp_mode.data().asnumpy()[0]
            filter_weights = self.filter_weights.data()

            bias = None if self.bias is None else self.bias.data()

            if mode == 0:
                conv_out = mx.nd.Convolution(data = data,
                                  weight = filter_weights,
                                  bias = bias,
                                  no_bias = (bias is None),
                                  kernel = self._kernel,
                                  stride = self._stride,
                                  pad = self._pad,
                                  num_filter = filter_weights.shape[0],
                                  layout = self._layout)

                return conv_out
            else:
                sp_data_weights = self.sp_data_weights.data()
                sp_out_weights = self.sp_out_weights.data()

                if self._clip_weights:
                    with mx.autograd.pause():
                        self.sp_data_weights.set_data(mx.nd.clip(data=sp_data_weights, a_min=-self._scale_quant, a_max=self._scale_quant))
                        self.sp_out_weights.set_data(mx.nd.clip(data=sp_out_weights, a_min=-self._scale_quant, a_max=self._scale_quant))
                        sp_data_weights = self.sp_data_weights.data()
                        sp_out_weights = self.sp_out_weights.data()

                if mode == 2:
                    '''
                    Quantize weights multiplying the data and the output of the hidden
                    layer to values {-1, 0, 1}. The ter_quant operator passes gradients
                    straight through.
                    '''

                    sp_data_weights = mx.nd.Custom(sp_data_weights, scale=self._scale_quant, op_type=self._quant_op)
                    sp_out_weights = mx.nd.Custom(sp_out_weights, scale=self._scale_quant, op_type=self._quant_op)

                elif mode == 3:
                    sp_data_weights = mx.nd.Custom(sp_data_weights, scale=self._scale_quant, op_type=self._quant_op)
                elif mode == 4:
                    sp_out_weights = mx.nd.Custom(sp_out_weights, scale=self._scale_quant, op_type=self._quant_op)

                ### Adjust padding if out_patch > 1, so that the output dimension can be made large enough by deconvolution below (with target_shape)
                pad_x = self._pad[0] + ceil((data.shape[2]%self._sp_data_weight_stride[0])/2)
                pad_y = self._pad[1] + ceil((data.shape[3]%self._sp_data_weight_stride[1])/2)
                sp_data_mul = mx.nd.Convolution(data = data,weight = sp_data_weights,bias = None,no_bias = True,kernel = self._sp_data_weight_kernel,stride = self._sp_data_weight_stride,pad = (pad_x, pad_y),num_group = self._num_group,num_filter = self._nbr_mul,layout = self._layout)

                sp_data_mul_norm = mx.nd.BatchNorm(data=sp_data_mul,
                                              gamma=self.sp_batchnorm_gamma.data(),
                                              beta=self.sp_batchnorm_beta.data(),
                                              moving_mean=self.sp_batchnorm_running_mean.data(),
                                              moving_var=self.sp_batchnorm_running_var.data(),
                                              fix_gamma=True)

                with mx.autograd.pause():
                    self.sp_in_weights.set_data(mx.nd.clip(data=self.sp_in_weights.data(), a_min=0.0, a_max=100.0))
                sp_mul = mx.nd.multiply(self.sp_in_weights.data(), sp_data_mul_norm)

                deconv_targ_x = data.shape[2]//self._stride[0] if self._pad[0] > 0 else (data.shape[2] - self._kernel[0] + 1)//self._stride[0]
                deconv_targ_y = data.shape[3]//self._stride[1] if self._pad[1] > 0 else (data.shape[3] - self._kernel[1] + 1)//self._stride[1]
                sp_out = mx.nd.Deconvolution(data = sp_mul,
                                  weight = sp_out_weights,
                                  bias = bias,
                                  no_bias = (self.bias is None),
                                  kernel = self._out_patch,
                                  stride = self._out_patch,
                                  target_shape = (deconv_targ_x, deconv_targ_y),
                                  num_filter = filter_weights.shape[0],
                                  layout = self._layout)

                return sp_out




class SumProdDense(mx.gluon.HybridBlock):

    def __init__(self, nbr_mul, units, in_units, use_bias=True, use_batchnorm=True, scale_quant=1.0, clip_weights=False, rescale_quant=False, prefix=None, **kwargs):
        super(SumProdDense, self).__init__(prefix=prefix, **kwargs)

        self._nbr_mul = nbr_mul
        self._units = units
        self._in_units = in_units
        self._scale_quant = scale_quant
        self._clip_weights = clip_weights
        self._quant_op = 'ter_quant_rescaled' if rescale_quant else 'ter_quant'
        self._use_batchnorm = use_batchnorm

        self.weights = self.params.get('weight', shape=(units, in_units))
        self.bias = self.params.get('bias', shape=(units,), init=mx.init.Constant(0)) if use_bias else None

        self.sp_in_weights = self.params.get('sp_in_weights', shape=(1,nbr_mul))
        self.sp_data_weights = self.params.get('sp_data_weights', shape=(nbr_mul, in_units))
        self.sp_out_weights = self.params.get('sp_out_weights', shape=(units, nbr_mul))

        ## batchnorm
        if use_batchnorm:
            self.sp_batchnorm_gamma = self.params.get('sp_batchnorm_gamma', shape=(1,self._nbr_mul,1), init=mx.init.One())
            self.sp_batchnorm_beta = self.params.get('sp_batchnorm_beta', shape=(1,self._nbr_mul,1), init=mx.init.Zero())
            self.sp_batchnorm_running_mean = self.params.get('sp_batchnorm_running_mean', shape=(1,self._nbr_mul,1))
            self.sp_batchnorm_running_var = self.params.get('sp_batchnorm_running_var', shape=(1,self._nbr_mul,1))

        self.sp_mode = self.params.get('sp_mode', shape=(1,), dtype=np.uint8)


    def forward(self, data):
        with data.context:
            mode = self.sp_mode.data().asnumpy()[0]

            bias = None if self.bias is None else self.bias.data()


            if mode == 0:
                dense_out = mx.nd.FullyConnected(data=data,
                                  weight=self.weights.data(),
                                  bias=bias,
                                  num_hidden=self._units,
                                  no_bias=(bias is None))

                return dense_out
            else:
                sp_data_weights = self.sp_data_weights.data()
                sp_out_weights = self.sp_out_weights.data()

                if self._clip_weights:
                    with mx.autograd.pause():
                        self.sp_data_weights.set_data(mx.nd.clip(data=sp_data_weights, a_min=-self._scale_quant, a_max=self._scale_quant))
                        self.sp_out_weights.set_data(mx.nd.clip(data=sp_out_weights, a_min=-self._scale_quant, a_max=self._scale_quant))
                        sp_data_weights = self.sp_data_weights.data()
                        sp_out_weights = self.sp_out_weights.data()

                if mode == 2:
                    sp_data_weights = mx.nd.Custom(sp_data_weights, scale=self._scale_quant, op_type=self._quant_op)
                    sp_out_weights = mx.nd.Custom(sp_out_weights, scale=self._scale_quant, op_type=self._quant_op)

                elif mode == 3:
                    sp_data_weights = mx.nd.Custom(sp_data_weights, scale=self._scale_quant, op_type=self._quant_op)
                elif mode == 4:
                    sp_out_weights = mx.nd.Custom(sp_out_weights, scale=self._scale_quant, op_type=self._quant_op)

                sp_data_mul = mx.nd.FullyConnected(data=data,
                                  weight=sp_data_weights,
                                  bias=None,
                                  num_hidden=self._nbr_mul,
                                  no_bias=True)

                if self._use_batchnorm:
                    sp_data_mul_norm = mx.nd.BatchNorm(data=sp_data_mul,
                                                  gamma=self.sp_batchnorm_gamma.data(),
                                                  beta=self.sp_batchnorm_beta.data(),
                                                  moving_mean=self.sp_batchnorm_running_mean.data(),
                                                  moving_var=self.sp_batchnorm_running_var.data(),
                                                  fix_gamma=True)
                else:
                    sp_data_mul_norm = sp_data_mul

                with mx.autograd.pause():
                    self.sp_in_weights.set_data(mx.nd.clip(data=self.sp_in_weights.data(), a_min=0.0, a_max=100.0))
                sp_mul = mx.nd.multiply(self.sp_in_weights.data(), sp_data_mul_norm)

                sp_out = mx.nd.FullyConnected(data=sp_mul,
                                  weight=sp_out_weights,
                                  bias=bias,
                                  num_hidden=self._units,
                                  no_bias=(bias is None))

                return sp_out


class SumProdPlain(mx.gluon.HybridBlock):

    def __init__(self, nbr_mul, units, in_units_A, in_units_B, use_bias=False, scale_quant=1.0, clip_weights=False, rescale_quant=False, prefix=None, **kwargs): #, use_batchnorm=False
        super(SumProdPlain, self).__init__(prefix=prefix, **kwargs)

        self._nbr_mul = nbr_mul
        self._units = units
        self._scale_quant = scale_quant
        self._clip_weights = clip_weights
        self._quant_op = 'ter_quant_rescaled' if rescale_quant else 'ter_quant'

        self.bias = self.params.get('bias', shape=(units,)) if use_bias else None

        self.sp_weights_A = self.params.get('sp_weights_A', shape=(nbr_mul, in_units_A))
        self.sp_weights_B = self.params.get('sp_weights_B', shape=(nbr_mul, in_units_B))
        self.sp_weights_C = self.params.get('sp_weights_C', shape=(units, nbr_mul))

        self.scale_A = self.params.get('sp_scale_A', shape=(1,1))
        self.scale_B = self.params.get('sp_scale_B', shape=(1,1))
        self.scale_C = self.params.get('sp_scale_C', shape=(1,1))

        self.sp_mode = self.params.get('sp_mode', shape=(1,), dtype=np.uint8)


    def forward(self, data):
        with data[0].context:
            mode = self.sp_mode.data().asnumpy()[0]
            data_A = data[0]
            data_B = data[1]
            bias = None if self.bias is None else self.bias.data()

            if mode == 0:
                sp_out = mx.nd.FullyConnected(lhs=data_A, rhs=data_B)

                if bias is not None:
                    sp_out = sp_out + bias

                return sp_out
            else:
                sp_weights_A = self.sp_weights_A.data()
                sp_weights_B = self.sp_weights_B.data()
                sp_weights_C = self.sp_weights_C.data()

                if self._clip_weights:
                    with mx.autograd.pause():
                        self.sp_weights_A.set_data(mx.nd.clip(data=sp_weights_A, a_min=-self._scale_quant, a_max=self._scale_quant))
                        self.sp_weights_B.set_data(mx.nd.clip(data=sp_weights_B, a_min=-self._scale_quant, a_max=self._scale_quant))
                        self.sp_weights_C.set_data(mx.nd.clip(data=sp_weights_C, a_min=-self._scale_quant, a_max=self._scale_quant))
                        sp_weights_A = self.sp_weights_A.data()
                        sp_weights_B = self.sp_weights_B.data()
                        sp_weights_C = self.sp_weights_C.data()

                if mode == 2:
                    sp_weights_A = mx.nd.Custom(sp_weights_A, scale=self._scale_quant, op_type=self._quant_op)
                    sp_weights_B = mx.nd.Custom(sp_weights_B, scale=self._scale_quant, op_type=self._quant_op)
                    sp_weights_C = mx.nd.Custom(sp_weights_C, scale=self._scale_quant, op_type=self._quant_op)

                sp_in_A_mul = mx.nd.FullyConnected(data=data_A,
                                  weight=sp_weights_A,
                                  bias=None,
                                  num_hidden=self._nbr_mul,
                                  no_bias=True)
                sp_in_A_mul = mx.nd.multiply(self.scale_A.data(), sp_in_A_mul)

                sp_in_B_mul = mx.nd.FullyConnected(data=data_B,
                                  weight=sp_weights_B,
                                  bias=None,
                                  num_hidden=self._nbr_mul,
                                  no_bias=True)
                sp_in_B_mul = mx.nd.multiply(self.scale_B.data(), sp_in_B_mul)

                sp_mul = mx.nd.multiply(sp_in_A_mul, sp_in_B_mul)

                sp_out = mx.nd.FullyConnected(data=sp_mul,
                                  weight=sp_weights_C,
                                  bias=bias,
                                  num_hidden=self._units,
                                  no_bias=(bias is None))
                sp_out = mx.nd.multiply(self.scale_C.data(), sp_out)

                return sp_out


class SumProdLSTM(mx.gluon.Block):

    def __init__(self, nbr_mul, units, num_layers=1, dropout=0, layout='NTC', input_size=None, scale_quant=1.0, clip_weights=False, rescale_quant=False, prefix='splstm_', **kwargs):
        super(SumProdLSTM, self).__init__(prefix=prefix, **kwargs)

        self._units = units
        self._input_size = input_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._layout = layout

        self._scale_quant = scale_quant
        self._clip_weights = clip_weights
        self._quant_op = 'ter_quant_rescaled' if rescale_quant else 'ter_quant'
        self.sp_mode = self.params.get('sp_mode', shape=(1,), dtype=np.uint8)


        self.i2h_sp_in_weights = []
        self.i2h_sp_data_weights = []
        self.i2h_sp_out_weights = []
        self.h2h_sp_in_weights = []
        self.h2h_sp_data_weights = []
        self.h2h_sp_out_weights = []

        self.i2h_bias = []
        self.h2h_bias = []

        input_size_l = input_size

        for l in range(num_layers):
            self.i2h_sp_in_weights.append(
                [self.params.get('l%i_%i_i2h_sp_in_weights'%(l, i), shape=(nbr_mul, 1), init=mx.init.Constant(1.0)) for i in range(4)]
            )
            self.i2h_sp_data_weights.append(
                [self.params.get('l%i_%i_i2h_sp_data_weights'%(l, i), shape=(nbr_mul, input_size_l)) for i in range(4)]
            )
            self.i2h_sp_out_weights.append(
                [self.params.get('l%i_%i_i2h_sp_out_weights'%(l, i), shape=(units, nbr_mul)) for i in range(4)]
            )
            self.h2h_sp_in_weights.append(
                [self.params.get('l%i_%i_h2h_sp_in_weights'%(l, i), shape=(nbr_mul, 1), init=mx.init.Constant(1.0)) for i in range(4)]
            )
            self.h2h_sp_data_weights.append(
                [self.params.get('l%i_%i_h2h_sp_data_weights'%(l, i), shape=(nbr_mul, units)) for i in range(4)]
            )
            self.h2h_sp_out_weights.append(
                [self.params.get('l%i_%i_h2h_sp_out_weights'%(l, i), shape=(units, nbr_mul)) for i in range(4)]
            )
            self.i2h_bias.append(
                self.params.get('l%i_i2h_bias'%l, shape=(4*units,),
                                init=mx.init.Zero()))
            self.h2h_bias.append(
                self.params.get('l%i_h2h_bias'%l, shape=(4*units,),
                                init=mx.init.Zero()))
            input_size_l = units

    def begin_state(self, batch_size=0, func=mx.nd.zeros, **kwargs):
        states = [func(name='%sh0_%d'%(self.prefix, i), shape=(2, batch_size, self._units), **kwargs)
            for i in range(self._num_layers)]
        return states

    def construct_weight_matrix(self, sp_in_weights_param, sp_data_weights_param, sp_out_weights_param):
        sp_data_weights = sp_data_weights_param.data()
        sp_out_weights = sp_out_weights_param.data()

        mode = self.sp_mode.data().asnumpy()[0]

        if mode == 0:
            raise NotImplementedError("Mode 0 not implemented for SumProdLSTM!")

        if self._clip_weights:
            with mx.autograd.pause():
                sp_data_weights_param.set_data(mx.nd.clip(data=sp_data_weights, a_min=-self._scale_quant, a_max=self._scale_quant))
                sp_out_weights_param.set_data(mx.nd.clip(data=sp_out_weights, a_min=-self._scale_quant, a_max=self._scale_quant))
                sp_data_weights = sp_data_weights_param.data()
                sp_out_weights = sp_out_weights_param.data()

        if mode == 2:
            sp_data_weights = mx.nd.Custom(sp_data_weights, scale=self._scale_quant, op_type=self._quant_op)
            sp_out_weights = mx.nd.Custom(sp_out_weights, scale=self._scale_quant, op_type=self._quant_op)

        elif mode == 3:
            sp_data_weights = mx.nd.Custom(sp_data_weights, scale=self._scale_quant, op_type=self._quant_op)
        elif mode == 4:
            sp_out_weights = mx.nd.Custom(sp_out_weights, scale=self._scale_quant, op_type=self._quant_op)


        with mx.autograd.pause():
            sp_in_weights_param.set_data(mx.nd.clip(data=sp_in_weights_param.data(), a_min=0.0, a_max=100.0))
        sp_in_weights = sp_in_weights_param.data()

        sp_mtx = mx.nd.dot(sp_out_weights,
                  mx.nd.multiply(sp_in_weights, sp_data_weights))

        return sp_mtx


    def forward(self, inputs, states):
        if self._layout == 'NTC':
            inputs = mx.nd.swapaxes(inputs, dim1=0, dim2=1)

        i2h_weight = []
        h2h_weight = []

        for l in range(self._num_layers):
            i2h_weight_l = []
            h2h_weight_l = []
            for i in range(4):
                i2h_weight_l.append(self.construct_weight_matrix(
                    self.i2h_sp_in_weights[l][i],
                    self.i2h_sp_data_weights[l][i],
                    self.i2h_sp_out_weights[l][i]
                ))
                h2h_weight_l.append(self.construct_weight_matrix(
                    self.h2h_sp_in_weights[l][i],
                    self.h2h_sp_data_weights[l][i],
                    self.h2h_sp_out_weights[l][i]
                ))
            i2h_weight.append(mx.nd.concat(*i2h_weight_l, dim=0))
            h2h_weight.append(mx.nd.concat(*h2h_weight_l, dim=0))


        ctx = inputs.context
        params = sum(zip(i2h_weight, h2h_weight), ())
        i2h_bias_data = [self.i2h_bias[i].data() for i in range(self._num_layers)]
        h2h_bias_data = [self.h2h_bias[i].data() for i in range(self._num_layers)]
        params += sum(zip(i2h_bias_data, h2h_bias_data), ())
        params = (i.reshape((-1,)) for i in params)
        params = mx.nd.concat(*params, dim=0)

        rnn = mx.nd.RNN(inputs, params, *states, state_size=self._units,
                          num_layers=self._num_layers, bidirectional=False,
                          p=self._dropout, state_outputs=True, mode='lstm')

        outputs, states = rnn[0], [rnn[1], rnn[2]]

        if self._layout == 'NTC':
            outputs = mx.nd.swapaxes(outputs, dim1=0, dim2=1)

        return outputs, states
