import keras
from keras.engine.training import Model
from keras.optimizers import Adam


def construct_model():

    # model = Model(input=inputs, output=pred)

    adam = Adam()
    # model.compile(loss='categorical_crossentropy', optimizer=adam)

    # return model