# coding:utf-8
import string
import numpy as np
import pandas as pd
from random import shuffle
import math

from pandas.core.frame import DataFrame

UNKNSYM = u'ξ'
NOSYM = u'ℵ'


def encode_data(x, maxlen, vocab, vocab_size, check):
    """
    Iterate over the loaded data and create a matrix of size maxlen x vocabsize

    This is then placed in a 3D matrix of size data_samples x maxlen x vocab_size.
    Each character is encoded into a one-hot array. Chars not in the vocab
    are encoded into an all zero vector.
    """

    input_data = np.zeros((len(x), maxlen, vocab_size))

    for dix, sent in enumerate(x):

        counter = 0
        sent_array = np.zeros((maxlen, vocab_size))

        try:
            chars = list(sent.lower())  # .replace(' ', ''))
        except:
            print("ERROR " + str(dix) + " " + str(sent))
            continue

        for c in chars:
            if counter >= maxlen:
                break
            else:
                char_array = np.zeros(vocab_size, dtype=np.int)
                if c in check:
                    ix = vocab[c]
                    char_array[ix] = 1
                else:
                    # char not in set, we replace it with special symbol
                    ix = vocab[UNKNSYM]
                    char_array[ix] = 1

                sent_array[counter, :] = char_array
                counter += 1

        input_data[dix, :, :] = sent_array

    return input_data


def create_vocab_set():
    alphabet = (list(u"йцукенгшщзхъёфывапролджэячсмитьбю") +
                list(string.ascii_lowercase) + list(string.digits) +
                list(string.punctuation) + ['\n'])
    vocab_size = len(alphabet)
    check = set(alphabet)

    vocab = {}
    reverse_vocab = {}

    for ix, t in enumerate(alphabet):
        vocab[t] = ix
        reverse_vocab[ix] = t

    return vocab, reverse_vocab, vocab_size, check


def prepare_relations(filepath, splitting=0.9):
    all_data_list = pd.read_csv(filepath, header=None, encoding="utf-8", sep="\t")
    all_data_list.dropna()

    # shuffle(all_data_list)

    splitting = int(math.floor(splitting * len(all_data_list)))
    train_ds = DataFrame(all_data_list[:splitting])
    test_ds = DataFrame(all_data_list[splitting:])

    train_ds.to_csv('data/train.csv', encoding="utf-8", index=False, header=False, sep=",", quotechar='"')
    test_ds.to_csv('data/test.csv', encoding="utf-8", index=False, header=False, sep=",", quotechar='"')


def load_relations():
    train = pd.read_csv('data/train.csv', header=None, encoding="utf-8")
    train = train.dropna()
    x_train = np.array(train.ix[:, 0:2])
    y_train = np.array(train.ix[:, 3])

    test = pd.read_csv('data/test.csv', header=None, encoding="utf-8")
    test = test.dropna()
    x_test = np.array(test.ix[:, 0:2])
    y_test = np.array(test.ix[:, 3])

    return (x_train, y_train), (x_test, y_test)


def mini_batch_generator(x, y, vocab, vocab_size, vocab_check, maxlen, batch_size):

    for i in range(0, len(x), batch_size):

        x_sample = x[i:i + batch_size]
        y_sample = y[i:i + batch_size]

        input_data0 = encode_data(x_sample[:, 0], maxlen, vocab, vocab_size, vocab_check)
        input_data1 = encode_data(x_sample[:, 1], maxlen, vocab, vocab_size, vocab_check)
        input_data2 = encode_data(x_sample[:, 2], maxlen, vocab, vocab_size, vocab_check)

        input_data = np.concatenate([input_data0, input_data1, input_data2], axis=2)

        y_for_fitting = encode_data(y_sample, maxlen, vocab, vocab_size, vocab_check)

        yield (input_data, y_for_fitting, x_sample, y_sample)


def decode_data(matrix, reverse_vocab):
    """
        data_samples x maxlen x vocab_size
    """
    try:
        return "".join(
            [reverse_vocab[np.argmax(row)] for encoded_matrix in matrix for row in encoded_matrix]).strip(
            NOSYM)
    except:
        return "ERROR"

# def shuffle_matrix(x, y):
#     stacked = np.hstack((np.matrix(x).T, np.matrix(y).T))
#     np.random.shuffle(stacked)
#     xi = np.array(stacked[:, 0]).flatten()
#     yi = np.array(stacked[:, 1]).flatten()
#     return xi, yi

if __name__ == '__main__':
    prepare_relations("data/relations.pairs.test.tsv")
