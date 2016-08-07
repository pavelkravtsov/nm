# coding:utf-8
import string
import numpy as np


def encode_data(sent, maxlen, vocab, vocab_size, check):
    """
    Iterate over the loaded data and create a matrix of size maxlen x vocabsize

    This is then placed in a 3D matrix of size data_samples x maxlen x vocab_size.
    Each character is encoded into a one-hot array. Chars not in the vocab
    are encoded into an all zero vector.
    """

    counter = 0
    sent_array = np.zeros((maxlen, vocab_size))
    chars = list(sent.lower().replace(' ', ''))

    for c in chars:
        if counter >= maxlen:
            pass
        else:
            char_array = np.zeros(vocab_size, dtype=np.int)

            if c in check:
                ix = vocab[c]
                char_array[ix] = 1
            sent_array[counter, :] = char_array
            counter += 1

    return sent_array


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


def read_encoded_relations(filepath):
    vocab, reverse_vocab, vocab_size, check = create_vocab_set()
    with open(filepath, "r+") as input_file:
        for line in input_file:
            line = line.strip()
            c4 = []
            for word in line.split("\t"):
                encode_data(word, 20, vocab, len(vocab), check)