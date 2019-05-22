from keras.layers import Conv1D, Conv2D, MaxPool2D, MaxPool1D, Dropout, Activation, GlobalAvgPool1D
from keras.layers import GlobalMaxPool1D, GlobalMaxPool1D, Flatten, Dense, Input, BatchNormalization
from keras.layers import Add, Lambda
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.activations import relu, softmax
from keras.backend import squeeze,stack, expand_dims


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

def conv1d_vgg(input_shape,n_classes):
    
    X_input = Input(shape = input_shape)
    
    X = Conv1D(8,9,activation = relu,padding = 'same')(X_input)
    X = BatchNormalization()(X)
    X = MaxPool1D(4)(X)
    
    X = Conv1D(16,9,activation = relu,padding = 'same')(X)
    X = BatchNormalization()(X)
    X = MaxPool1D(4)(X)
    
    X = Conv1D(32,9,activation = relu, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Conv1D(32,9,activation = relu, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = MaxPool1D(2)(X)
    
    X = Conv1D(64,9,activation = relu, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Conv1D(64,9,activation = relu, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = MaxPool1D(2)(X)
    
    X = Conv1D(128,9,activation = relu, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Conv1D(128,9,activation = relu, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = MaxPool1D(2)(X)
    
    X = Conv1D(256,9,activation = relu, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Conv1D(256,9,activation = relu, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = MaxPool1D(2)(X)
    
    X = Conv1D(512,9,activation = relu, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Conv1D(512,9,activation = relu, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = MaxPool1D(2)(X)
    
    X = Conv1D(1024,9,activation = relu, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Conv1D(1024,9,activation = relu, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = MaxPool1D(2)(X)
    
    X = Flatten()(X)
    
    X = Dense(256,activation=relu)(X)
    X = Dropout(0.5)(X)
    X = Dense(128,activation=relu)(X)
    X = Dropout(0.5)(X)
    X = Dense(n_classes,activation=softmax)(X)
    
    model = Model(inputs = X_input, outputs = X)
    
    return model

def identity_block(X,N):
    X_1 = Conv1D(4*N,9,padding='same')(X)
    
    X_2 = Conv1D(N,9,padding='same')(X)
    X_2 = BatchNormalization()(X_2)
    X_2 = Activation(relu)(X_2)
    
    X_2 = Conv1D(N,9,padding='same')(X)
    X_2 = BatchNormalization()(X_2)
    X_2 = Activation(relu)(X_2)
    
    X_2 = Conv1D(4*N,9,padding='same')(X)
    X_2 = BatchNormalization()(X_2)
    
    X = Add()([X_1,X_2])
    X = Activation(relu)(X)
    
    return X

def conv1d_resnet(input_shape,n_classes):
    
    X_input = Input(shape = input_shape)
    
    X = Conv1D(48,80,strides = 4, padding = 'same')(X_input)
    X = BatchNormalization()(X)
    X = Activation(relu)(X)
    X = MaxPool1D(4)(X)
    
    X = identity_block(X,48)
    X = identity_block(X,48)
    X = identity_block(X,48)
    
    X = MaxPool1D(4)(X)
    
    X = identity_block(X,96)
    X = identity_block(X,96)
    X = identity_block(X,96)
    X = identity_block(X,96)
    
    X = MaxPool1D(4)(X)
    
    X = identity_block(X,192)
    X = identity_block(X,192)
    X = identity_block(X,192)
    X = identity_block(X,192)
    X = identity_block(X,192)
    X = identity_block(X,192)
    
    X = MaxPool1D(4)(X)
    
    X = identity_block(X,384)
    X = identity_block(X,384)
    X = identity_block(X,384)
    
    X = GlobalAvgPool1D()(X)
    
    X = Dense(n_classes,activation=softmax)(X)
    
    model = Model(inputs = X_input, outputs = X)
    
    return model

def conv1d_resnet_v2(input_shape,n_classes):
    
    X_input = Input(shape = input_shape)
    
    X = Conv1D(48,80,strides = 4, padding = 'same')(X_input)
    X = BatchNormalization()(X)
    X = Activation(relu)(X)
    X = MaxPool1D(4)(X)
    
    X = identity_block(X,48)
    X = identity_block(X,48)
    
    X = MaxPool1D(4)(X)
    
    X = identity_block(X,96)
    X = identity_block(X,96)
    
    X = MaxPool1D(4)(X)
    
    X = identity_block(X,192)
    X = identity_block(X,192)
    
    X = MaxPool1D(4)(X)
    
    X = identity_block(X,384)
    X = identity_block(X,384)
    
    X = GlobalAvgPool1D()(X)
    
    X = Dense(n_classes,activation=softmax)(X)
    
    model = Model(inputs = X_input, outputs = X)
    
    return model