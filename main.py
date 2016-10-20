# coding:utf-8
"""
    Character-level attempt to build recurrent neural network architecture
    for building correct morphological forms by analogy
"""

import datetime
import logging
import numpy as np
import data_helpers
import model_all_stacked

# setting logging

logging.basicConfig(filename='all_results.log',
                    format='[%(asctime)s] [%(levelname)s] %(message)s',
                    level=logging.DEBUG)

lg = logging.getLogger("L")
lg.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
lg.addHandler(ch)

# for reproducibility

np.random.seed(0123)

# setting parameters
# todo: argparse

subset = None
save = False
model_name_path = 'params/model.json'
model_weights_path = 'params/model_weights.h5'

latent_dimension = 125
maxlen = 25

batch_size = 80
test_batch_size = 20

nb_epoch = 10

lg.info('Loading data...')

# loading triples and predicted words
(x_train, y_train), (x_test, y_test) = data_helpers.load_relations()

# building vocabs for chars
vocab, reverse_vocab, vocab_size, check = data_helpers.create_vocab_set()

lg.info('Build model...')

# building model for predicting words
model = model_all_stacked.construct_model(maxlen, vocab_size * 3, vocab_size, latent_dimension)

lg.info('Fit model...')
initial = datetime.datetime.now()

# nb_epoch times we scan data trying to learn morphology-by-analogy
for e in xrange(nb_epoch):

    xi, yi = x_train, y_train
    xi_test, yi_test = x_test, y_test

    if subset:
        batches = data_helpers.mini_batch_generator(xi[:subset], yi[:subset],
                                                    vocab, vocab_size, check, maxlen,
                                                    batch_size=batch_size)
    else:
        batches = data_helpers.mini_batch_generator(xi, yi,
                                                    vocab, vocab_size, check, maxlen,
                                                    batch_size=batch_size)

    test_batches = data_helpers.mini_batch_generator(xi_test, yi_test, vocab,
                                                     vocab_size, check, maxlen,
                                                     batch_size=test_batch_size)

    loss = 0.0
    step = 1
    start = datetime.datetime.now()

    lg.info('-------- epoch {} --------'.format(e))

    for x_train, y_train, x_text_tr, y_text_tr in batches:

        f = model.train_on_batch(x_train, y_train)
        loss += f
        loss_avg = loss / step

        if step % 10 == 0:
            lg.info('- TRAINING step {} \t loss {}'.format(step, loss_avg))
        step += 1

    test_loss = 0.0
    test_step = 0

    lg.info(" -- TESTING NOW -- ")

    for x_test_batch, y_test_batch, x_text, y_text in test_batches:
        f_ev = model.test_on_batch(x_test_batch, y_test_batch)
        test_loss += f_ev
        test_loss_avg = test_loss / test_step
        test_step += 1

        lg.info('- TESTING step {}\tloss {}'.format(test_step, test_loss_avg))
        predicted_seq = model.predict(np.array([x_test_batch[0]]))
        lg.info(
            'Shapes x {} y_true {} y_pred {}'.format(
                x_test_batch[0].shape,
                y_test_batch[0].shape,
                predicted_seq[0].shape))
        lg.info(u'Input:       \t[' + "|".join(map(lambda x:x[:maxlen], list(x_text[0]))) + "] -> ? ")
        lg.info(u'Expected:    \t[' + y_text[0] + "]")
        lg.info(u'Predicted: \t[' + data_helpers.decode_data(predicted_seq, reverse_vocab) + "]")
        lg.info('----------------------------------------------------------------')

    stop = datetime.datetime.now()
    e_elap = stop - start
    t_elap = stop - initial
    lg.info('Epoch {}. Loss: {}\nEpoch time: {}. Total time: {}\n'.format(e, test_loss, e_elap, t_elap))

    # if save:
    #     print('Saving model params...')
    #     json_string = model.to_json()
    #     with open(model_name_path, 'w') as f:
    #         json.dump(json_string, f)
    #
    #     model.save_weights(model_weights_path)
