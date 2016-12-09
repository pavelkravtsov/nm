# coding:utf-8

from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam


def construct_model(maxlen, output_dimension, n_hidden, optimizer):
    """
    Склеены три слова
    """
    input_dimension = 3 * output_dimension

    input = Input(shape=(maxlen, input_dimension), name='input')
    lstm_encode = LSTM(n_hidden, activation='relu')(input)

    encoded_copied = RepeatVector(n=maxlen)(lstm_encode)
    lstm_decode = LSTM(output_dim=output_dimension, return_sequences=True,
                       activation='softmax')(encoded_copied)

    encoder = Model(input, lstm_decode)
    encoder.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return encoder


def construct_simpliest_model(maxlen, output_dimension, n_hidden, n_layers, optimizer):
    """
    Простейшая LSTM последовательность -> последовательность
    """
    input_dimension = 3 * output_dimension

    lstm_encode = Input(shape=(maxlen, input_dimension), name='input')
    for _ in xrange(n_layers - 1):
        lstm_encode = LSTM(n_hidden, return_sequences=True,
                           activation="relu")(lstm_encode)
    lstm_encode = LSTM(output_dimension, return_sequences=True,
                       activation="softmax")(lstm_encode)

    encoder = Model(input, lstm_encode)
    encoder.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return encoder


def construct_complex_model(maxlen, output_dimension, n_hidden, optimizer):
    """
    Каждое слово LSTM-> в вектор -> конкатенировать вектора LSTM-> выход
    """
    input_dimension = output_dimension

    input1 = Input(shape=(maxlen, input_dimension))
    input2 = Input(shape=(maxlen, input_dimension))
    input3 = Input(shape=(maxlen, input_dimension))

    lstm_encode1 = LSTM(n_hidden, activation='relu')(input1)
    lstm_encode2 = LSTM(n_hidden, activation='relu')(input2)
    lstm_encode3 = LSTM(n_hidden, activation='relu')(input3)

    encode = merge([lstm_encode1, lstm_encode2, lstm_encode3], mode="concat")
    hidden_copied = RepeatVector(n=maxlen)(encode)
    out = LSTM(output_dim=output_dimension, return_sequences=True,
               activation='softmax')(hidden_copied)

    encoder = Model([input1, input2, input3], out)
    encoder.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return encoder


def construct_generatorlike_model(maxlen, output_dimension, sub_length, n_hidden, optimizer):
    """
    Каждое слово LSTM-> в вектор -> конкатенировать вектора - concat LSTM-> выход
                              sub_length предыдущих символов/
    """
    input_dimension = output_dimension

    input1 = Input(shape=(maxlen, input_dimension))
    input2 = Input(shape=(maxlen, input_dimension))
    input3 = Input(shape=(maxlen, input_dimension))
    input4 = Input(shape=(sub_length, input_dimension))

    lstm_encode1 = LSTM(n_hidden, activation='relu', return_sequences=False)(input1)
    lstm_encode2 = LSTM(n_hidden, activation='relu', return_sequences=False)(input2)
    lstm_encode3 = LSTM(n_hidden, activation='relu', return_sequences=False)(input3)

    encode = merge([lstm_encode1, lstm_encode2, lstm_encode3], mode="concat")

    hidden_copied = RepeatVector(n=sub_length)(encode)
    # (sub_length, 3 * n_hidden + input_dimension)
    merged = merge([hidden_copied, input4], mode="concat", concat_axis=2)
    merged = TimeDistributed(Dense(n_hidden, activation='relu'))(merged)
    out = LSTM(output_dim=output_dimension, return_sequences=True,
               activation='softmax', input_length=sub_length)(merged)

    encoder = Model([input1, input2, input3, input4], out)
    encoder.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return encoder


def construct_cnn_model(maxlen, output_dimension,
                        nb_filter, n_hidden, max_filter_size,
                        optimizer):
    """
    Каждое слово CNN-> в вектор -> конкатенировать вектора LSTM-> выход
    """
    input_dimension = output_dimension

    input1 = Input(shape=(maxlen, input_dimension))
    input2 = Input(shape=(maxlen, input_dimension))
    input3 = Input(shape=(maxlen, input_dimension))

    lengths = list(range(1, max_filter_size + 1))
    encode1 = merge([Flatten()(Convolution1D(nb_filter * l, l)(input1)) for l in lengths], mode="concat")
    encode2 = merge([Flatten()(Convolution1D(nb_filter * l, l)(input2)) for l in lengths], mode="concat")
    encode3 = merge([Flatten()(Convolution1D(nb_filter * l, l)(input3)) for l in lengths], mode="concat")

    encode = merge([encode1, encode2, encode3], mode="concat")
    decode = Dense(n_hidden)(encode)
    out = Reshape((maxlen, output_dimension))(Dense((maxlen * output_dimension))(decode))

    model = Model([input1, input2, input3], out)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model
