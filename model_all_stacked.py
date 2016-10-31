# coding:utf-8

from keras.layers import Input, LSTM
from keras.layers.core import RepeatVector
from keras.layers.recurrent import SimpleRNN
from keras.models import Model
from keras.optimizers import Adam


def construct_model(maxlen, input_dimension, output_dimension, lstm_vector_output_dim, lr=0.001):
    """
    Склеены три слова
    """
    input = Input(shape=(maxlen, input_dimension), name='input')

    lstm_encode = LSTM(lstm_vector_output_dim, activation='relu')(input)
    #lstm_encode = SimpleRNN(lstm_vector_output_dim, activation='relu')(input)

    encoded_copied = RepeatVector(n=maxlen)(lstm_encode)

    lstm_decode = LSTM(output_dim=output_dimension, return_sequences=True, activation='softmax')(encoded_copied)
    #lstm_decode = SimpleRNN(output_dim=output_dimension, return_sequences=True, activation='softmax')(encoded_copied)

    encoder = Model(input, lstm_decode)

    adam = Adam(lr=lr)
    encoder.compile(loss='categorical_crossentropy', optimizer=adam)

    return encoder


def construct_simpliest_model(maxlen, input_dimension, output_dimension, lstm_vector_output_dim):
    """
    Простейшая RNN последовательность -> последовательность
    """
    input = Input(shape=(maxlen, input_dimension), name='input')

    lstm_encode = SimpleRNN(output_dimension, return_sequences=True)(input)

    encoder = Model(input, lstm_encode)

    adam = Adam()
    encoder.compile(loss='mse', optimizer=adam)

    return encoder

def construct_multilayer_model(maxlen, input_dimension, output_dimension, lstm_vector_output_dim, lr=0.001, clipnorm=10.0):
    """
    Склеены три слова
    """
    input = Input(shape=(maxlen, input_dimension), name='input')

    lstm_encode = input
    lstm_encode = LSTM(lstm_vector_output_dim, return_sequences=True, activation='relu')(lstm_encode)
    lstm_encode = LSTM(lstm_vector_output_dim, return_sequences=True, activation='relu')(lstm_encode)
    lstm_encode = LSTM(lstm_vector_output_dim, activation='relu')(lstm_encode)
    
    encoded_copied = RepeatVector(n=maxlen)(lstm_encode)

    lstm_decode = encoded_copied
    lstm_decode = LSTM(output_dim=output_dimension, return_sequences=True, activation='relu')(lstm_decode)
    lstm_decode = LSTM(output_dim=output_dimension, return_sequences=True, activation='relu')(lstm_decode)
    lstm_decode = LSTM(output_dim=output_dimension, return_sequences=True, activation='softmax')(lstm_decode)
    
    encoder = Model(input, lstm_decode)

    adam = Adam(lr=lr, clipnorm=clipnorm)
    encoder.compile(loss='categorical_crossentropy', optimizer=adam)

    return encoder
