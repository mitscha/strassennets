import mxnet as mx
from mxnet import gluon, autograd
from CRNNmodel import LSTMCharWord, SumProdLSTMCharWord
from BatchLoaderUnk import BatchLoaderUnk, Tokens
import numpy as np

import argparse
import time
import math

# logging
import os
import logging
import tempfile


def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [v.detach() for v in hidden]
    else:
        hidden = hidden.detach()
    return hidden

def eval_model(model, loader, batch_size, split_index, context):
    total_L = 0.0
    ntotal = 0
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    loader.reset_batch_pointer(split_index)
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)
    for i in range(0, loader.split_sizes[split_index]):
        batch = next(loader.next_batch(split_index))
        output, hidden = model(mx.nd.array(batch[0]['chars'], ctx=context), hidden)
        target = mx.nd.flatten(mx.nd.array(batch[1],ctx=context))
        L = loss(output, target)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal

def train(model, lr, lr_decay, batch_size, epochs, context, loader, grad_clip, seq_len, decay_delta, log_interval, save_path, teacher_model=None, dist_lambda=0.0, dist_temp=1.0):
    prev_ppl = float("Inf")

    trainer = gluon.Trainer(model.collect_params(), 'sgd',
                            {'learning_rate': lr,
                             'momentum': 0,
                             'wd': 0})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    dist_loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)

    for epoch in range(epochs):
        total_L = 0.0
        start_time = time.time()
        hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)

        if teacher_model is not None:
            teacher_hidden = teacher_model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)

        for ibatch in range(loader.split_sizes[0]):
            batch = next(loader.next_batch(0))
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(mx.nd.array(batch[0]['chars'], ctx=context), hidden)
                target = mx.nd.flatten(mx.nd.array(batch[1],ctx=context))
                if teacher_model is None:
                    L = loss(output, target)
                else:
                    teacher_output, teacher_hidden = teacher_model(mx.nd.array(batch[0]['chars'], ctx=context), teacher_hidden)
                    L = (1-dist_lambda) * loss(output, target) + dist_lambda * (dist_temp**2) * dist_loss(output/dist_temp, mx.nd.softmax(teacher_output/dist_temp))
                L.backward()

            grads = [v.grad(context) for v in model.collect_params().values() if v.grad_req != 'null']
            # Here gradient is for the whole batch.
            # So we multiply max_norm by batch_size and bptt size to balance it.
            gluon.utils.clip_global_norm(grads, grad_clip * seq_len * batch_size) # * seq_len  * batch_size

            trainer.step(batch_size)
            total_L += mx.nd.sum(L).asscalar()

            if ibatch % log_interval == 0 and ibatch > 0:
                cur_L = total_L / batch_size / log_interval
                logging.info('[Epoch %d Batch %d] loss %.2f, ppl %.2f'%(
                    epoch, ibatch, cur_L, math.exp(cur_L)))
                total_L = 0.0

        val_L = eval_model(model, loader, batch_size, 1, context)
        val_ppl = math.exp(val_L)

        logging.info('[Epoch %d] time cost %.2fs, valid loss %.2f, valid ppl %.2f'%(
            epoch, time.time()-start_time, val_L, val_ppl))

        if val_ppl > prev_ppl - decay_delta:
            lr = lr*lr_decay
            trainer.set_learning_rate(lr)
        prev_ppl = val_ppl

    test_L = eval_model(model, loader, batch_size, 2, context)
    logging.info('test loss %.2f, test ppl %.2f'%(test_L, math.exp(test_L)))
    if save_path is not None:
        model.collect_params().save(save_path)

def get_param_by_key(net, key1, key2=''):
    params = []
    for k, v in net.collect_params().items():
        if k.find(key1)!=-1 and k.find(key2)!=-1:
            params.append(v)

    return params

def set_mode_by_key(net, key, mode):
    for p in get_param_by_key(net, key, 'sp_mode'):
        p.set_data(mx.nd.reshape(mode*mx.nd.ones((1,), dtype=np.uint8), (1,)))

def set_sp_train_by_key(net, key, train=True):
    for p in get_param_by_key(net, key, 'sp_in_weights'): p.grad_req = 'write' if train else 'null'
    for p in get_param_by_key(net, key, 'sp_data_weights'): p.grad_req = 'write' if train else 'null'
    for p in get_param_by_key(net, key, 'sp_out_weights'): p.grad_req = 'write' if train else 'null'
    for p in get_param_by_key(net, key, 'sp_batchnorm_gamma'): p.grad_req = 'write' if train else 'null'
    for p in get_param_by_key(net, key, 'sp_batchnorm_beta'): p.grad_req = 'write' if train else 'null'

def print_grads_modes(net, ctx):
    for k, v in net.collect_params().items():
        logging.info('%s: %s'%(k, v.grad_req))
        if k.find('sp_mode')!=-1:
            logging.info('sp_mode: %i'%v.data(ctx).asnumpy()[0])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a word+character-level language model')
    # data
    parser.add_argument('--data_dir', type=str, default='data/ptb', help='data directory. Should contain train.txt/valid.txt/test.txt with input data')
    # model params
    parser.add_argument('--rnn_size', type=int, default=650, help='size of LSTM internal state')
    parser.add_argument('--highway_layers', type=int, default=2, help='number of highway layers')
    parser.add_argument('--word_vec_size', type=int, default=650, help='dimensionality of word embeddings')
    parser.add_argument('--char_vec_size', type=int, default=15, help='dimensionality of character embeddings')
    parser.add_argument('--feature_maps', type=int, nargs='+', default=[50,100,150,200,200,200,200], help='number of feature maps in the CNN')
    parser.add_argument('--kernels', type=int, nargs='+', default=[1,2,3,4,5,6,7], help='conv net kernel widths')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the LSTM')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout. 0 = no dropout')
    parser.add_argument('--nbr_mul', type=float, default=1.0, help='number of multiplications relative to the number of output channels')
    parser.add_argument('--out_nbr_mul', type=int, default=2000, help='number of multiplications for the decoder, for quant_mode=1,2,3,4')
    parser.add_argument('--quant_mode', type=int, default=0, help='0: no sum-prod, 1: full-precision sum-prod, 2: quantized sum-prod, 5: vanilla implementation, 6: sp_out_identity, 7: ternarized weights')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=2, help='starting learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.5, help='learning rate decay')
    parser.add_argument('--decay_when', type=float, default=1, help='decay if validation perplexity does not improve by more than this much')
    parser.add_argument('--seq_length', type=int, default=35, help='number of timesteps to unroll for')
    parser.add_argument('--batch_size', type=int, default=20, help='number of sequences to train on in parallel')
    parser.add_argument('--max_epochs', type=int, default=25, help='number of full passes through the training data')
    parser.add_argument('--max_grad_norm', type=float, default=5, help='normalize gradients at')
    parser.add_argument('--max_word_l', type=int, default=65, help='maximum word length')
    parser.add_argument('--epochs_cont', type=int, default=0, help='number of epochs of continued training with freezed ternary weights')
    parser.add_argument('--epochs_pre', type=int, default=0, help='number of epochs of full-precision pretraining')
    parser.add_argument('--learning_rate_cont', type=float, default=0.01, help='learning rate for continued training with freezed weights')
    parser.add_argument('--learning_rate_pre', type=float, default=0.01, help='learning rate for full-precision pretraining')
    parser.add_argument('--n_words', type=int, default=30000, help='max number of words in model')
    parser.add_argument('--n_chars', type=int, default=100, help='max number of char in model')

    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--print_every', type=int, default=50, help='how many steps/minibatches between printing out the loss')
    parser.add_argument('--log_dir', type=str, default='logs', help='folder to save log and model')
    parser.add_argument('--EOS', type=str, default='+', help='<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')

    # Distillation
    parser.add_argument('--teacher_model', type=str, default=None, help='Model for distillation')
    parser.add_argument('--dist_lambda', type=float, default=0.5, help='Weight for distillation loss term')
    parser.add_argument('--dist_temp', type=float, default=1.0, help='Temperature for distillation')

    # parse input params
    opt = parser.parse_args()

    # initialize logging
    if not os.path.exists(opt.log_dir):
          os.makedirs(opt.log_dir)
    logfile = next(tempfile._get_candidate_names())
    logging.basicConfig(level=logging.INFO, filename=os.path.join(opt.log_dir, logfile)+'.log')

    logging.info(opt)


    # global constants for certain tokens
    opt.tokens = Tokens(
        EOS=opt.EOS,
        UNK='|',    # unk word token
        START='{',  # start-of-word token
        END='}',    # end-of-word token
        ZEROPAD=' ' # zero-pad token
    )

    mx.random.seed(123)
    np.random.seed(321)

    loader = BatchLoaderUnk(opt.tokens, opt.data_dir, opt.batch_size, opt.seq_length, opt.max_word_l, opt.n_words, opt.n_chars)

    opt.word_vocab_size = min(opt.n_words, len(loader.idx2word))
    opt.char_vocab_size = min(opt.n_chars, len(loader.idx2char))
    opt.max_word_l = loader.max_word_l

    context = mx.gpu(opt.gpu_index)

    # Load distillation model if required
    if opt.teacher_model is not None and opt.quant_mode in (2,3,4,6):
        teacher_model = LSTMCharWord(vocab_size=opt.char_vocab_size,
                             word_vocab_size=opt.word_vocab_size,
                             num_embed=opt.char_vec_size,
                             ker_width=opt.kernels,
                             num_filters=opt.feature_maps,
                             word_len=opt.max_word_l,
                             hw_layers=opt.highway_layers,
                             lstm_layers=opt.num_layers,
                             lstm_units=opt.rnn_size,
                             dropout=0.0)
        teacher_model.collect_params().load(opt.teacher_model, ctx=context)
        teacher_model.collect_params().setattr('grad_req', 'null')
    else:
        teacher_model = None

    if opt.quant_mode == 5:

        model = LSTMCharWord(vocab_size=opt.char_vocab_size,
                             word_vocab_size=opt.word_vocab_size,
                             num_embed=opt.char_vec_size,
                             ker_width=opt.kernels,
                             num_filters=opt.feature_maps,
                             word_len=opt.max_word_l,
                             hw_layers=opt.highway_layers,
                             lstm_layers=opt.num_layers,
                             lstm_units=opt.rnn_size,
                             dropout=opt.dropout)

        model.collect_params().initialize(mx.init.Xavier(), ctx=context)

        logging.info(model)

        model.collect_params()['lstmcharword0_dense1_bias'].initialize(mx.init.Uniform(0.05), ctx=context, force_reinit=True)
        model.collect_params()['lstmcharword0_dense3_bias'].initialize(mx.init.Uniform(0.05), ctx=context, force_reinit=True)

        for k, _ in model.collect_params().items():
            logging.info(k)
    # 6: sp_out_weights = identity; 7: ternary quantization/TWN
    elif opt.quant_mode >= 6:
        model = SumProdLSTMCharWord(vocab_size=opt.char_vocab_size,
                             word_vocab_size=opt.word_vocab_size,
                             num_embed=opt.char_vec_size,
                             ker_width=opt.kernels,
                             num_filters=opt.feature_maps,
                             word_len=opt.max_word_l,
                             hw_layers=opt.highway_layers,
                             lstm_layers=opt.num_layers,
                             lstm_units=opt.rnn_size,
                             dropout=opt.dropout,
                             # only works for nbr_mul=1
                             nbr_mul=1,
                             # no batchnorm in sum-prod hidden layers for ternary quantization
                             dense_batchnorm=False if opt.quant_mode == 7 else True,
                             # need to adapt out dimension as sp_out_weights is set to identity
                             out_nbr_mul=opt.word_vocab_size)

        for p in get_param_by_key(model, ''):
            p.grad_req = 'null'
        set_sp_train_by_key(model, '', True)
        get_param_by_key(model, 'embedding0')[0].grad_req = 'write'

        for p in get_param_by_key(model, 'bias'): p.grad_req = 'write'
        for p in get_param_by_key(model, 'beta'): p.grad_req = 'write'
        for p in get_param_by_key(model, 'gamma'): p.grad_req = 'write'

        ### Don't quantize sp_out weights, but keep them constant at identity
        for p in get_param_by_key(model, 'sp_mode'):
            p.initialize(init=mx.init.Constant(3), ctx=context)
            p.grad_req = 'null'

        # In ternary quantization case, fix sp_in_weights to ones (inefficient but quick)
        if opt.quant_mode == 7:
            for p in get_param_by_key(model, 'sp_in_weights'):
                p.initialize(init=mx.init.One(), ctx=context)
                p.grad_req = 'null'
        ###

        model.collect_params().initialize(mx.init.Xavier(), ctx=context)

        model.collect_params()['sumprodlstmcharword0_sumproddense1_bias'].initialize(mx.init.Uniform(0.05), ctx=context, force_reinit=True)
        model.collect_params()['sumprodlstmcharword0_sumproddense3_bias'].initialize(mx.init.Uniform(0.05), ctx=context, force_reinit=True)

        ###
        for p in get_param_by_key(model, 'sp_out_weights'):
            assert p.data().shape[0] == p.data().shape[1]
            idmtx = mx.nd.zeros_like(p.data(), ctx=p.data().context)
            for i in range(idmtx.shape[0]):
                if(len(idmtx.shape) == 4):
                    idmtx[i,i,0,0] = 1.0
                else:
                    idmtx[i,i] = 1.0
            p.set_data(idmtx)
            p.grad_req = 'null'
        ###

    else:

        model = SumProdLSTMCharWord(vocab_size=opt.char_vocab_size,
                             word_vocab_size=opt.word_vocab_size,
                             num_embed=opt.char_vec_size,
                             ker_width=opt.kernels,
                             num_filters=opt.feature_maps,
                             word_len=opt.max_word_l,
                             hw_layers=opt.highway_layers,
                             lstm_layers=opt.num_layers,
                             lstm_units=opt.rnn_size,
                             dropout=opt.dropout,
                             nbr_mul=opt.nbr_mul,
                             dense_batchnorm=True,
                             out_nbr_mul=opt.out_nbr_mul)

        for p in get_param_by_key(model, 'sp_mode'):
            p.initialize(init=mx.init.Constant(opt.quant_mode), ctx=context, force_reinit=True)
            p.grad_req = 'null'

        model.collect_params().initialize(mx.init.Xavier(), ctx=context)

        model.collect_params()['sumprodlstmcharword0_sumproddense1_bias'].initialize(mx.init.Uniform(0.05), ctx=context, force_reinit=True)
        model.collect_params()['sumprodlstmcharword0_sumproddense3_bias'].initialize(mx.init.Uniform(0.05), ctx=context, force_reinit=True)

        if opt.quant_mode == 1 or opt.quant_mode == 2:
            for p in get_param_by_key(model, ''):
                p.grad_req = 'null'

            set_sp_train_by_key(model, '', True)

            get_param_by_key(model, 'embedding0')[0].grad_req = 'write'

            for p in get_param_by_key(model, 'bias'): p.grad_req = 'write'

            for p in get_param_by_key(model, 'highway', 'beta'):
                p.grad_req = 'write'

            for p in get_param_by_key(model, 'highway', 'gamma'):
                p.grad_req = 'write'
        else:
            set_sp_train_by_key(model, '', False)
            for p in get_param_by_key(model, 'running_mean'):
                p.grad_req = 'null'

            for p in get_param_by_key(model, 'running_var'):
                p.grad_req = 'null'


        for p in get_param_by_key(model, 'conv', 'sp_out_weights'):
            idmtx = mx.nd.zeros_like(p.data(), ctx=p.data().context)
            for i in range(idmtx.shape[0]):
                idmtx[i,i,0,0] = 1.0
            p.set_data(idmtx)
            p.grad_req = 'null'

    if opt.quant_mode in (2,3,4,6) and opt.epochs_pre > 0:
        set_mode_by_key(model, '', 1)
        print_grads_modes(model, context)
        train(model=model,
              lr=opt.learning_rate_pre,
              lr_decay=opt.learning_rate_decay,
              batch_size=opt.batch_size,
              epochs=opt.epochs_pre,
              context=context,
              loader=loader,
              grad_clip=opt.max_grad_norm,
              seq_len=opt.seq_length,
              decay_delta=opt.decay_when,
              log_interval=opt.print_every,
              save_path=None,
              teacher_model=teacher_model,
              dist_lambda=opt.dist_lambda,
              dist_temp=opt.dist_temp)
        set_mode_by_key(model, '', opt.quant_mode)

    print_grads_modes(model, context)

    train(model=model,
          lr=opt.learning_rate,
          lr_decay=opt.learning_rate_decay,
          batch_size=opt.batch_size,
          epochs=opt.max_epochs,
          context=context,
          loader=loader,
          grad_clip=opt.max_grad_norm,
          seq_len=opt.seq_length,
          decay_delta=opt.decay_when,
          log_interval=opt.print_every,
          save_path=os.path.join(opt.log_dir, logfile)+'_model.params',
          teacher_model=teacher_model,
          dist_lambda=opt.dist_lambda,
          dist_temp=opt.dist_temp)

    if opt.quant_mode in (2,3,4,6) and opt.epochs_cont > 0:
        for p in get_param_by_key(model, 'sp_data_weights'):
            p.grad_req = 'null'
        for p in get_param_by_key(model, 'sp_out_weights'):
            p.grad_req = 'null'
        print_grads_modes(model, context)
        train(model=model,
              lr=opt.learning_rate_cont,
              lr_decay=opt.learning_rate_decay,
              batch_size=opt.batch_size,
              epochs=opt.epochs_cont,
              context=context,
              loader=loader,
              grad_clip=opt.max_grad_norm,
              seq_len=opt.seq_length,
              decay_delta=0,
              log_interval=opt.print_every,
              save_path=os.path.join(opt.log_dir, logfile)+'_model_cont.params',
              teacher_model=teacher_model,
              dist_lambda=opt.dist_lambda,
              dist_temp=opt.dist_temp)
