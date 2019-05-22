from tensorflow.keras.layers import Conv1D, Conv2D, MaxPool2D, MaxPool1D, Dropout, Activation, GlobalAvgPool1D
from tensorflow.keras.layers import GlobalMaxPool1D, GlobalMaxPool2D, Flatten, Dense, Input, BatchNormalization
from tensorflow.keras.layers import Add, ZeroPadding2D, AvgPool2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.activations import relu, softmax


def malley_cnn_40(input_shape, n_classes):

    X_input = Input(input_shape)
    
    X = Conv2D(64,[3,7],padding = 'same')(X_input)
    X = Activation('relu')(X)
    X = MaxPool2D([2,1])(X)
    
    X = Conv2D(128,[7,1],padding = 'same')(X)
    X = Activation('relu')(X)
    X = MaxPool2D([4,1])(X)
    
    X = Conv2D(256,[5,1],padding = 'valid')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(512,[1,7],padding = 'same')(X)
    X = Activation('relu')(X)
    
    X = GlobalMaxPool2D()(X)
    
    X = Dense(512,activation = 'relu')(X)
    X = Dropout(0.5)(X)
    
    X = Dense(n_classes, activation = 'softmax')(X)

    model = Model(inputs = X_input, outputs = X)
    
    return model

def malley_cnn_80(input_shape, n_classes):

    X_input = Input(input_shape)
    
    X = Conv2D(64,[3,7],padding = 'same')(X_input)
    X = Activation('relu')(X)
    X = MaxPool2D([2,1])(X)
    
    X = Conv2D(128,[7,1],padding = 'same')(X)
    X = Activation('relu')(X)
    X = MaxPool2D([4,1])(X)
    
    X = Conv2D(256,[10,1],padding = 'valid')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(512,[1,7],padding = 'same')(X)
    X = Activation('relu')(X)
    
    X = GlobalMaxPool2D()(X)
    
    X = Dense(512,activation = 'relu')(X)
    X = Dropout(0.5)(X)
    
    X = Dense(n_classes, activation = 'softmax')(X)

    model = Model(inputs = X_input, outputs = X)
    
    return model


def malley_cnn_120(input_shape, n_classes):

    X_input = Input(input_shape)
    
    X = Conv2D(64,[3,7],padding = 'same')(X_input)
    X = Activation('relu')(X)
    X = MaxPool2D([3,1])(X)
    
    X = Conv2D(128,[7,1],padding = 'same')(X)
    X = Activation('relu')(X)
    X = MaxPool2D([4,1])(X)
    
    X = Conv2D(256,[10,1],padding = 'valid')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(512,[1,7],padding = 'same')(X)
    X = Activation('relu')(X)
    
    X = GlobalMaxPool2D()(X)
    
    X = Dense(512,activation = 'relu')(X)
    X = Dropout(0.5)(X)
    
    X = Dense(n_classes, activation = 'softmax')(X)

    model = Model(inputs = X_input, outputs = X)
    
    return model


def cnn_trad_fpool3_40(input_shape, n_classes):

    X_input = Input(input_shape)
    
    X = Conv2D(64,[8,20],padding = 'valid')(X_input)
    X = Activation(relu)(X)
    X = MaxPool2D([3,1])(X)
    
    X = Conv2D(64,[4,10],padding = 'valid')(X)
    X = Activation(relu)(X)
    
    X = Flatten()(X)
    
    X = Dense(32)(X)
    X = Dense(128)(X)
    X = Dropout(0.5)(X)
    X = Activation(relu)(X)
    
    X = Dense(n_classes)(X)
    X = Activation(softmax)(X)

    model = Model(inputs = X_input, outputs = X)
    
    return model


def cnn_trad_fpool3_120(input_shape, n_classes):

    X_input = Input(input_shape)
    
    X = Conv2D(64,[24,20],padding = 'valid')(X_input)
    X = Activation(relu)(X)
    X = MaxPool2D([9,1])(X)
    
    X = Conv2D(64,[4,10],padding = 'valid')(X)
    X = Activation(relu)(X)
    
    X = Flatten()(X)
    
    X = Dense(32)(X)
    X = Dense(128)(X)
    X = Dropout(0.5)(X)
    X = Activation(relu)(X)
    
    X = Dense(n_classes)(X)
    X = Activation(softmax)(X)

    model = Model(inputs = X_input, outputs = X)
    
    return model


def cnn_one_fstride4_40(input_shape, n_classes):

    X_input = Input(input_shape)
    
    X = Conv2D(186,[8,32],padding = 'valid',strides=[4,1])(X_input)
    X = Activation(relu)(X)
    
    X = Flatten()(X)
    
    X = Dense(32)(X)
    
    X = Dense(128)(X)
    X = Dropout(0.5)(X)
    X = Activation(relu)(X)
    
    X = Dense(128)(X)
    X = Dropout(0.5)(X)
    X = Activation(relu)(X)
    
    X = Dense(n_classes)(X)
    X = Activation(softmax)(X)

    model = Model(inputs = X_input, outputs = X)
    
    return model


def cnn_one_fstride4_120(input_shape, n_classes):

    X_input = Input(input_shape)
    
    X = Conv2D(186,[24,32],padding = 'valid',strides=[12,1])(X_input)
    X = Activation(relu)(X)
    
    X = Flatten()(X)
    
    X = Dense(32)(X)
    
    X = Dense(128)(X)
    X = Dropout(0.5)(X)
    X = Activation(relu)(X)
    
    X = Dense(128)(X)
    X = Dropout(0.5)(X)
    X = Activation(relu)(X)
    
    X = Dense(n_classes)(X)
    X = Activation(softmax)(X)

    model = Model(inputs = X_input, outputs = X)
    
    return model
