import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn, rnn

import sys
sys.path.append('../common/')
from sumprodlayer import SumProdLSTM, SumProd2DConv, SumProdDense


class CharCNN(gluon.Block):
    def __init__(self, ker_width, num_embed, num_filters, word_len, conv_class=nn.Conv2D, **kwargs):
        super(CharCNN, self).__init__(**kwargs)
        self._num_fil = len(num_filters)

        for i, cfil in enumerate(zip(num_filters, ker_width)):
            convmax = nn.Sequential()
            convmax.add(conv_class(cfil[0], (cfil[1], num_embed), in_channels=1))
            convmax.add(nn.Activation('tanh'))
            convmax.add(nn.MaxPool2D((word_len-cfil[1]+1,1)))
            setattr(self, 'fil%i'%i, convmax)

    def forward(self, x):
        output = []
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        xres = x.reshape((batch_size*seq_len, 1, x.shape[2], x.shape[3]))
        for i in range(self._num_fil):
            convmaxi = getattr(self, 'fil%i'%i)
            yi = convmaxi(xres)
            output.append(mx.nd.reshape(yi, (batch_size, seq_len, -1)))
        y = mx.nd.concat(*output, dim=2)
        return y

class HighwayLayer(gluon.Block):
    def __init__(self, num_units, dense_class=nn.Dense, prefix='highway', **kwargs):
        super(HighwayLayer, self).__init__(prefix=prefix, **kwargs)
        self.dense = nn.Sequential()
        self.dense.add(dense_class(num_units, in_units=num_units))
        self.dense.add(nn.Activation('relu'))

        self.transform = nn.Sequential()
        self.transform.add(dense_class(num_units, in_units=num_units))
        self.transform.add(nn.Activation('sigmoid'))

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        xres = x.reshape((batch_size*seq_len, x.shape[2]))
        transform_in = self.transform(xres - 2.0)
        y = transform_in * self.dense(xres) + (1.0 - transform_in) * xres
        return mx.nd.reshape(y, (batch_size, seq_len, -1))

class Decoder(gluon.HybridBlock):
    def __init__(self, out_units, in_units, dense_class=nn.Dense, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dense = dense_class(out_units, in_units=in_units) #, flatten=False

    def hybrid_forward(self, F, x):
        orig_shape = x.shape
        xres = x.reshape((x.shape[0]*x.shape[1], x.shape[2]))
        return mx.nd.reshape(self.dense(xres), (orig_shape[0], orig_shape[1],-1))

class LSTMCharWord(gluon.Block):
    def __init__(self, vocab_size, word_vocab_size, num_embed, ker_width, num_filters, word_len, hw_layers, lstm_layers, lstm_units, dropout, **kwargs):
        super(LSTMCharWord, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.Sequential()
            self.features.add(nn.Embedding(vocab_size, num_embed))

            hw_units = num_embed
            if (ker_width is not None) and (num_filters is not None):
                self.features.add(CharCNN(ker_width, num_embed, num_filters, word_len))
                hw_units = sum(num_filters)

            if hw_layers is not None:
                for _ in range(hw_layers):
                    self.features.add(HighwayLayer(hw_units))

            self.rnn = rnn.LSTM(lstm_units, num_layers=lstm_layers, layout='NTC', dropout=dropout,
                                input_size=hw_units)

            self.drop = nn.Dropout(dropout)

            self.decoder = Decoder(word_vocab_size, lstm_units)

    def forward(self, inputs, hidden):
        features = self.features(inputs)
        rnn_out, hidden = self.rnn(features, hidden)
        decoded = self.decoder(self.drop(rnn_out))
        return decoded, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


class SumProdLSTMCharWord(gluon.Block):
    def __init__(self, vocab_size, word_vocab_size, num_embed, ker_width, num_filters, word_len, hw_layers, lstm_layers, lstm_units, dropout, nbr_mul=1, dense_batchnorm=True, out_nbr_mul=0, **kwargs):
        super(SumProdLSTMCharWord, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.Sequential()
            self.features.add(nn.Embedding(vocab_size, num_embed))

            hw_units = num_embed

            def conv_class(channels, kernel, in_channels): return SumProd2DConv(channels, (channels, in_channels, kernel[0], kernel[1]), 'conv', kernel=kernel, pad=(0,0), rescale_quant=True, no_bias=False)
            if (ker_width is not None) and (num_filters is not None):
                self.features.add(CharCNN(ker_width, num_embed, num_filters, word_len, conv_class=conv_class))
                hw_units = sum(num_filters)

            def dense_class_hw(num_units, in_units): return SumProdDense(int(num_units*nbr_mul), num_units, in_units, use_batchnorm=dense_batchnorm, rescale_quant=True)
            if hw_layers is not None:
                for l in range(hw_layers):
                    self.features.add(HighwayLayer(hw_units, dense_class=dense_class_hw))
                    self.features.add(nn.BatchNorm(axis=2, in_channels=hw_units, prefix='highway_l%i_'%l))

            self.rnn = SumProdLSTM(int(lstm_units*nbr_mul), lstm_units, num_layers=lstm_layers, layout='NTC', dropout=dropout,
                                input_size=hw_units, rescale_quant=True)

            self.drop = nn.Dropout(dropout)

            def dense_class_dec(num_units, in_units):
                if out_nbr_mul < 1:
                    return SumProdDense(int(in_units*nbr_mul), num_units, in_units, use_batchnorm=dense_batchnorm, rescale_quant=True)
                else:
                    return SumProdDense(out_nbr_mul, num_units, in_units, use_batchnorm=dense_batchnorm, rescale_quant=True)

            self.decoder = Decoder(word_vocab_size, lstm_units, dense_class=dense_class_dec)

    def forward(self, inputs, hidden):
        features = self.features(inputs)
        rnn_out, hidden = self.rnn(features, hidden)
        decoded = self.decoder(self.drop(rnn_out))
        return decoded, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
