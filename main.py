
# coding: utf-8

from __future__ import division
from __future__ import print_function

import datetime
import logging
import json
import argparse

import numpy as np

import data_helpers
import model_all_stacked

from keras.optimizers import Adam
from random import choice, randrange
from functools import partial

# ## Настройка логгера

logging.basicConfig(format='%(message)s',
                     #format='[%(asctime)s] [%(levelname)s] %(message)s',
                     filename='log\all_results.log',
                     level=logging.DEBUG)

lg = logging.getLogger("L")
lg.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
ch.setFormatter(formatter)
lg.addHandler(ch)

# ## Парсинг аргументов

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int,
                    default=50,
                    help='default=50; epochs count')

parser.add_argument('--maxlen', type=int,
                    default=10,
                    help='default=10; max sequence length')

parser.add_argument('--gpu_fraction', type=float,
                    default=0.2,
                    help='default=0.2; GPU fraction, please, use with care')

parser.add_argument('--model_type',
                    default='simpliest',
                    choices=['simpliest', 'simple', 'complex', 'generatorlike', 'cnn'],
                    help='type of neural architecture')

parser.add_argument('--show_step',
                    default=10,
                    type=int,
                    help="show step")

parser.add_argument('--break_step',
                    default=1000,
                    type=int,
                    help="break step")

parser.add_argument('--n_layers',
                    default=1,
                    type=int,
                    help='number of layers')

parser.add_argument('--n_hidden',
                    default=64,
                    type=int,
                    help='number of hidden neurals')

parser.add_argument('--clipnorm',
                    default=0,
                    type=int,
                    help="clipnorm")

parser.add_argument('--activation',
                    default="relu",
                    choices=["relu", "sigmoid"])

args = parser.parse_args()


# ## Установка параметров

np.random.seed(0123)  # for reproducibility

train_file = "data/train.csv"
test_file = "data/test.csv"

maxlen = args.maxlen
batch_size = 100
test_batch_size = 20
nb_epoch = args.epochs
show_step = args.show_step
break_step = args.break_step
inf = 1000000

model_type = args.model_type   # number of model
clipnorm = args.clipnorm

N_LAYERS = args.n_layers
n_hidden = args.n_hidden
SUB_LENGTH = 3 # for generatorlike
nb_filter = 30
max_filter_size = 5
activation = args.activation

# ## Загрузка модели

lg.info('Prepare vocab...')
vocab, reverse_vocab, vocab_size, check = data_helpers.create_vocab_set()

lg.info('Build model...')
optimizer = Adam(clipnorm=clipnorm) if clipnorm else Adam()

if model_type == 'simple':
    mini_batch_generator = data_helpers.simple_batch_generator
    predict = lambda x, n: model.predict(np.array([x[n]]))
    model = model_all_stacked.construct_model(maxlen, vocab_size,
                                              n_hidden,
                                              optimizer)
elif model_type == 'simpliest':
    mini_batch_generator = data_helpers.simple_batch_generator
    predict = lambda x, n: model.predict(np.array([x[n]]))
    model = model_all_stacked.construct_simpliest_model(maxlen, vocab_size,
                                                        n_hidden, N_LAYERS,
                                                        optimizer, activation)
elif model_type == 'complex':
    mini_batch_generator = data_helpers.complex_batch_generator
    predict = lambda x, n: model.predict([np.array([x[0][n]]),
                                          np.array([x[1][n]]),
                                          np.array([x[2][n]])])
    model = model_all_stacked.construct_complex_model(maxlen, vocab_size,
                                                      n_hidden,
                                                      optimizer)
elif model_type == 'generatorlike':
    mini_batch_generator = partial(data_helpers.generatorlike_batch_generator,
                                   sub_length=SUB_LENGTH)
    def predict(x, n):
        return None
    
    model = model_all_stacked.construct_generatorlike_model(maxlen, vocab_size,
                                                            SUB_LENGTH, n_hidden,
                                                            optimizer)
elif model_type == 'cnn':
    mini_batch_generator = data_helpers.complex_batch_generator
    predict = lambda x, n: model.predict([np.array([x[0][n]]),
                                          np.array([x[1][n]]),
                                          np.array([x[2][n]])])
    model = model_all_stacked.construct_cnn_model(maxlen, vocab_size,
                                                      nb_filter, n_hidden, max_filter_size,
                                                      optimizer)
else:
    raise NotImplementedError()

# ## Mainloop
def tensor_size(shape):
    res = 1
    for size in shape:
        res *= size
    return res


lg.info('Fit model...')
initial = datetime.datetime.now()
lg.info("model have {} parameters".format(sum(tensor_size(t.shape) for t in model.get_weights())))

def train(batches, break_num=inf):
    loss = 0.0
    step = 0
    for x_train, y_train, x_text_tr, y_text_tr in batches:
        if step % break_num == 0:
            yield
        f = model.train_on_batch(x_train, y_train)
        loss += f
        loss_avg = loss / step
        if step % show_step == 0:
            lg.info('TRAIN step {}\tloss {}\tavg {}'.format(step, f, loss_avg))
        step += 1

def test(break_num=inf):

    loss = 0.0
    step = 0
    while True:
        test_batches = mini_batch_generator(test_file,
                                    vocab, vocab_size, check, maxlen,
                                    batch_size=test_batch_size)

        for x_test_batch, y_test_batch, x_text, y_text in test_batches:
            if step % break_num == 0:
                yield
            f_ev = model.test_on_batch(x_test_batch, y_test_batch)
            loss += f_ev
            loss_avg = loss / step
            step += 1

            n = randrange(len(x_test_batch))
            predicted_seq = predict(x_test_batch, n)
            # lg.info('TEST step {}\tloss {}\t avg {}'.format(step, f_ev, loss_avg))
            # lg.info('Shapes x {} y_true {} y_pred {}'.format(
            #         x_test_batch[0].shape,
            #         y_test_batch[0].shape,
            #         predicted_seq[0].shape))
            #lg.info(u'Input:     \t[' + u" | ".join(map(lambda x:x[:maxlen], list(x_text[n]))) + u"]")
            #lg.info(u'Expected:  \t[' + y_text[n] + u"]")
            #lg.info(u'Predicted: \t[' + data_helpers.decode_data(predicted_seq, reverse_vocab) + u"]")
            lg.info(predicted_seq)
            lg.info(x_test_batch[n])
            lg.info( u'[' + data_helpers.decode_data(predicted_seq, reverse_vocab) + u"]" + \
                    '\t[' + u" | ".join(map(lambda x:x[:maxlen], list(x_text[n]))) +  \
                    ' | ' + y_text[n] + u"]")
            lg.info('----------------------------------------------------------------')



for e in xrange(nb_epoch):
    lg.info('-------- epoch {} --------'.format(e))
    start = datetime.datetime.now()

    batches = mini_batch_generator(train_file,
                                   vocab, vocab_size, check, maxlen,
                                   batch_size=batch_size)

    
    test_iter = test(10)
    for _ in train(batches, break_step):
        next(test_iter)

    stop = datetime.datetime.now()
    e_elap = stop - start
    t_elap = stop - initial
    lg.info('Epoch {}. \nEpoch time: {}. Total time: {}\n'.format(e, e_elap, t_elap))

    # lg.info("Saving weights")
    # model_info = "{}_{}_{}".format(str(stop.date()), e, NUM)
    # model_weights_path = 'params/model_{}_weights.h5'.format(model_info)
    # model.save_weights(model_weights_path)

# ## Saving

# lg.info("Saving model")
# model_name_path = 'params/model_{}_.json'.format(model_info)
# json_string = model.to_json()
# with open(model_name_path, 'w') as f:
#     json.dump(json_string, f)
