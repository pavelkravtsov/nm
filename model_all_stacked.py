from keras.layers import Input, LSTM
from keras.layers.core import RepeatVector
from keras.models import Model
from keras.optimizers import Adam


def construct_model(maxlen, vocab_size, lstm_vector_output_dim):
    input = Input(shape=(maxlen, vocab_size), name='input', dtype='float32')

    lstm_encode = LSTM(lstm_vector_output_dim)(input)
    encoded_copied = RepeatVector(n=maxlen)(lstm_encode)
    lstm_decode = LSTM(output_dim=vocab_size, return_sequences=True, activation='softmax')(encoded_copied)

    encoder = Model(input, lstm_decode)

    adam = Adam()
    encoder.compile(loss='categorical_crossentropy', optimizer=adam)

    return encoder
