from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Activation, CuDNNGRU
from tensorflow.keras.layers import GRU, LSTM, TimeDistributed, Lambda, Dot, Softmax
from tensorflow.keras.layers import Conv2D, Conv1D, Reshape, Permute, GRUCell, LSTMCell
from tensorflow.keras.layers import Bidirectional, CuDNNLSTM, BatchNormalization, MaxPool1D
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.models import Model
from tensorflow.keras.backend import squeeze,stack, expand_dims

def CRNN1_1D(input_shape, n_classes):
    
    X_input = Input(input_shape)
    
    X = Lambda(lambda q: expand_dims(q, -1), name='expand_dims') (X_input)
    
    X = Conv1D(16,9, activation=relu, padding='valid')(X)
    X = MaxPool1D(8)(X)
    
    X = Conv1D(32,9,activation=relu,padding='valid')(X)
    X = MaxPool1D(8)(X)
    
    X = Conv1D(32,9,activation=relu,padding='valid')(X)
    X = MaxPool1D(6)(X)
        
    X = CuDNNGRU(32, return_sequences = True)(X)
    X = Dropout(0.1)(X)
    X = CuDNNGRU(32, return_sequences = True)(X)
    X = Dropout(0.1)(X)
    X = Flatten()(X)
    
    X = Dense(64)(X)
    X = Dropout(0.5)(X)
    X = Activation(relu)(X)
    
    X = Dense(n_classes, activation = softmax)(X)
    
    model = Model(inputs = X_input, outputs = X)
    
    return model

def CRNN_v1(input_shape, n_classes):
    
    X_input = Input(input_shape)
    
    X = Lambda(lambda q: expand_dims(q, -1), name='expand_dims') (X_input)
    
    X = Conv2D(32, kernel_size = [5,5], strides = [2,2],
               activation = relu, name = 'conv_1')(X_input)
    
    X = Conv2D(1,kernel_size=[1,1], strides = [1,1],
               activation = relu, name = 'conv_1x1')(X)
        
    X = Lambda(lambda q: squeeze(q, -1), name='squeeze_last_dim') (X)
        
    X = Permute((2,1)) (X)   
        
    X = CuDNNGRU(32, return_sequences = True)(X)
    
    X = CuDNNGRU(32, return_sequences = True)(X)
    
    X = Flatten()(X)
    
    X = Dense(64)(X)
    X = Dropout(0.5)(X)
    X = Activation(relu)(X)
    
    X = Dense(n_classes, activation = softmax)(X)
    
    model = Model(inputs = X_input, outputs = X)
    
    return model

def CRNN_v2(input_shape, n_classes):
    
    X_input = Input(input_shape)
    
    X = Permute((2,1,3))(X_input) 
    
    X = Conv2D(32, kernel_size = [5,5], strides = [2,2],
               activation = relu)(X)
    
    X = Conv2D(1,kernel_size=[1,1], strides = [1,1],
               activation = relu)(X)
        
    X = Lambda(lambda q: squeeze(q, -1), name='squeeze_last_dim') (X) 
        
    X = Bidirectional(CuDNNLSTM(32,return_sequences=True))(X)
    
    X = Bidirectional(CuDNNLSTM(32, return_sequences = True))(X)
    
    X = Flatten()(X)
    
    X = Dense(64)(X)
    X = Dropout(0.5)(X)
    X = Activation(relu)(X)
    
    X = Dense(n_classes, activation = softmax)(X)
    
    model = Model(inputs = X_input, outputs = X)
    
    return model

def AttRNNSpeechModelWave(input_shape, n_classes):
    
    X_input = Input(input_shape) 

    X = Lambda(lambda q: expand_dims(q, -1), name='expand_dims') (X_input)

    X = Conv1D(16,9, activation=relu, padding='valid')(X)
    X = MaxPool1D(8)(X)
    
    X = Conv1D(32,9,activation=relu,padding='valid')(X)
    X = MaxPool1D(8)(X)
    
    X = Conv1D(32,9,activation=relu,padding='valid')(X)
    X = MaxPool1D(6)(X)
    
    #X = Lambda(lambda q: squeeze(q, -1), name='squeeze_last_dim') (X) 

    X = Bidirectional(CuDNNLSTM(64, return_sequences = True)) (X) 
    X = Dropout(0.5)(X)
    X = Bidirectional(CuDNNLSTM(64, return_sequences = True)) (X) 
    X = Dropout(0.5)(X)
    
    xFirst = Lambda(lambda q: q[:,16]) (X) 
    query = Dense(128) (xFirst)
    query = Dropout(0.5)(query)
    
    attScores = Dot(axes=[1,2])([query, X]) 
    attScores = Softmax(name='attSoftmax')(attScores)

    attVector = Dot(axes=[1,1])([attScores, X])

    X = Dense(64, activation = 'relu')(attVector)
    X = Dropout(0.5)(X)
    X = Dense(32)(X)
    X = Dropout(0.5)(X)

    X = Dense(n_classes, activation = 'softmax', name='output')(X)

    model = Model(inputs = X_input, outputs = X)
    
    return model

def AttRNNSpeechModel(input_shape, n_classes):
    
    X_input = Input(input_shape) 

    X = Permute((2,1,3)) (X_input) 

    X = Conv2D(10, (5,1) , activation='relu', padding='same') (X)
    X = BatchNormalization() (X)
    X = Conv2D(1, (5,1) , activation='relu', padding='same') (X)
    X = BatchNormalization() (X)

    X = Lambda(lambda q: squeeze(q, -1), name='squeeze_last_dim') (X) 

    X = Bidirectional(CuDNNLSTM(64, return_sequences = True)) (X) 
    X = Dropout(0.5)(X)
    X = Bidirectional(CuDNNLSTM(64, return_sequences = True)) (X) 
    X = Dropout(0.5)(X)
    
    xFirst = Lambda(lambda q: q[:,16]) (X) 
    query = Dense(128) (xFirst)
    query = Dropout(0.5)(query)
    
    attScores = Dot(axes=[1,2])([query, X]) 
    attScores = Softmax(name='attSoftmax')(attScores)

    attVector = Dot(axes=[1,1])([attScores, X])

    X = Dense(64, activation = 'relu')(attVector)
    X = Dropout(0.5)(X)
    X = Dense(32)(X)
    X = Dropout(0.5)(X)

    X = Dense(n_classes, activation = 'softmax', name='output')(X)

    model = Model(inputs = X_input, outputs = X)
    
    return model