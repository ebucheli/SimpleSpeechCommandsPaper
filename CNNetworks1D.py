from tensorflow.keras.layers import Conv1D, Conv2D, MaxPool2D, MaxPool1D, Dropout, Activation, GlobalAvgPool1D
from tensorflow.keras.layers import GlobalMaxPool1D, GlobalMaxPool1D, Flatten, Dense, Input, BatchNormalization
from tensorflow.keras.layers import Add, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.backend import squeeze,stack, expand_dims


def conv1d_v1(input_shape,n_classes):

    X_input = Input(shape = input_shape)

    X = Lambda(lambda q: expand_dims(q, -1), name='expand_dims') (X_input)

    X = Conv1D(16,9,activation=relu,padding='valid')(X)
    X = Conv1D(16,9,activation=relu,padding='valid')(X)
    X = MaxPool1D(16)(X)
    X = Dropout(0.1)(X)

    X = Conv1D(32,3,activation=relu,padding='valid')(X)
    X = Conv1D(32,3,activation=relu,padding='valid')(X)
    X = MaxPool1D(4)(X)
    X = Dropout(0.1)(X)

    X = Conv1D(32,3,activation=relu,padding='valid')(X)
    X = Conv1D(32,3,activation=relu,padding='valid')(X)
    X = MaxPool1D(4)(X)
    X = Dropout(0.1)(X)

    X = Conv1D(256,3,activation=relu,padding='valid')(X)
    X = Conv1D(256,3,activation=relu,padding='valid')(X)
    X = GlobalMaxPool1D()(X)

    X = Dense(64,activation=relu)(X)
    X = Dense(128,activation=relu)(X)

    X = Dense(n_classes,activation=softmax)(X)

    model = Model(inputs = X_input,outputs = X)

    return model
