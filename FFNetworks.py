from tensorflow.keras.layers import Dropout, Activation, Flatten
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.activations import relu, softmax


def DNN_3HL(input_shape,n_classes):
    X_input = Input(input_shape)
    
    X = Flatten()(X_input)
    
    X = Dense(128,activation=relu)(X)
    X = Dropout(0.5)(X)
    X = Dense(128,activation=relu)(X)
    X = Dropout(0.5)(X)
    X = Dense(128,activation=relu)(X)
    X = Dropout(0.5)(X)
    
    X = Dense(n_classes,activation=softmax)(X)
    
    model = Model(inputs = X_input, outputs= X)
    
    return model
