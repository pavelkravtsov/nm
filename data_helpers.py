
# coding: utf-8

# In[1]:


import string
import numpy as np
from random import shuffle
import math
import codecs


# ## Спецсимволы

# In[11]:

# symbol we use to replace unknown symbols
UNKNSYM = u'*'
# symbol we use for padding
NOSYM = u'_'


# ## Кодирование слов в бинарные матрицы

# In[12]:

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

        counter = 0
        for c in chars:
            if counter >= maxlen:
                break
            else:
                char_array = np.zeros(vocab_size, dtype=np.int)
                if c in check:
                    ix = vocab[c]
                else:
                    ix = vocab[UNKNSYM]
                char_array[ix] = 1

                sent_array[counter, :] = char_array
                counter += 1

        char_array = np.zeros(vocab_size, dtype=np.int)
        char_array[vocab[NOSYM]] = 1
        while counter < maxlen:
            sent_array[counter, :] = char_array
            counter += 1
            
        input_data[dix, :, :] = sent_array

    return input_data


# ## Декодирование слов в бинарные матрицы

# In[13]:

def decode_data(matrix, reverse_vocab):
    """
        data_samples x maxlen x vocab_size
    """
    try:
        return "".join(
            [reverse_vocab[np.argmax(row)] for encoded_matrix in matrix for row in encoded_matrix])
    except:
        return "ERROR"


# ##  Словарь номеров символов и т. п.

# In[9]:

def create_vocab_set():
    alphabet = (list(NOSYM + u"йцукенгшщзхъёфывапролджэячсмитьбю-" + UNKNSYM))
    vocab_size = len(alphabet)
    check = set(alphabet)

    vocab = {}
    reverse_vocab = {}

    for ix, t in enumerate(alphabet):
        vocab[t] = ix
        reverse_vocab[ix] = t

    return vocab, reverse_vocab, vocab_size, check


# ## Генератор батчей из файла

# In[3]:

def simple_batch_generator(filename, vocab, vocab_size, vocab_check, maxlen, batch_size):
    with codecs.open(filename, "rt", encoding="utf-8") as data:
        run = True
        while run:
            x_sample = np.zeros((batch_size, 3), dtype=np.object)
            y_sample = np.zeros((batch_size,),   dtype=np.object)
            for i in range(batch_size):
                line = data.readline()
                if line:
                    words = line.strip("\n").split(",")
                    x_sample[i] = words[:3]
                    y_sample[i] = words[3]
                else:
                    run = False
                    break
            if not run:
                break

            input_data0 = encode_data(x_sample[:, 0], maxlen, vocab, vocab_size, vocab_check)
            input_data1 = encode_data(x_sample[:, 1], maxlen, vocab, vocab_size, vocab_check)
            input_data2 = encode_data(x_sample[:, 2], maxlen, vocab, vocab_size, vocab_check)
            
            input_data = np.concatenate([input_data0, input_data1, input_data2], axis=2)
            y_for_fitting = encode_data(y_sample, maxlen, vocab, vocab_size, vocab_check)

            yield (input_data, y_for_fitting, x_sample, y_sample)


# In[ ]:

def complex_batch_generator(filename, vocab, vocab_size, vocab_check, maxlen, batch_size):
    with codecs.open(filename, "rt", encoding="utf-8") as data:
        run = True
        while run:
            x_sample = np.zeros((batch_size, 3), dtype=np.object)
            y_sample = np.zeros((batch_size,),   dtype=np.object)
            for i in range(batch_size):
                line = data.readline()
                if line:
                    words = line.strip("\n").split(",")
                    x_sample[i] = words[:3]
                    y_sample[i] = words[3]
                else:
                    run = False
                    break
            if not run:
                break

            input_data0 = encode_data(x_sample[:, 0], maxlen, vocab, vocab_size, vocab_check)
            input_data1 = encode_data(x_sample[:, 1], maxlen, vocab, vocab_size, vocab_check)
            input_data2 = encode_data(x_sample[:, 2], maxlen, vocab, vocab_size, vocab_check)
            
            input_data = [input_data0, input_data1, input_data2]
            y_for_fitting = encode_data(y_sample, maxlen, vocab, vocab_size, vocab_check)

            yield (input_data, y_for_fitting, x_sample, y_sample)


# In[11]:

# def load_relations(train_file, test_file):
#     train = pd.read_csv(train_file, header=None, encoding="utf-8")
#     train = train.dropna()
#     x_train = np.array(train.ix[:, 0:2])
#     y_train = np.array(train.ix[:, 3])
#
#     test = pd.read_csv(test_file, header=None, encoding="utf-8")
#     test = test.dropna()
#     x_test = np.array(test.ix[:, 0:2])
#     y_test = np.array(test.ix[:, 3])
#
#     return (x_train, y_train), (x_test, y_test)


# In[14]:

# def shuffle_matrix(x, y):
#     stacked = np.hstack((np.matrix(x).T, np.matrix(y).T))
#     np.random.shuffle(stacked)
#     xi = np.array(stacked[:, 0]).flatten()
#     yi = np.array(stacked[:, 1]).flatten()
#     return xi, yi


# In[ ]:

# if __name__ == '__main__':
#     prepare_relations("data/relations.pairs.test.tsv")

