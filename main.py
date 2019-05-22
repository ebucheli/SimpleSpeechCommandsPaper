import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio
import librosa
import librosa.display

from SimpleSpeechCommands import get_word_dict, read_list, load_data
from SimpleSpeechCommands import append_examples,partition_directory
from ProcessAudio import normalize_waveforms
from Utilities import make_oh

from RNNetworks import CRNN1_1D
from tensorflow.keras.optimizers import Adam

def main(path_dataset):
    
    #Load dicts with commands and labels
    word_to_label,label_to_word = get_word_dict()
    
    
    # Set Sample Rate and file length
    sr = 16000
    file_length = 16000

    # Load filenames from previously generated lists

    training_files = read_list(path_dataset,'training_files.txt')
    validation_files = read_list(path_dataset,'validation_files.txt')
    testing_files = read_list(path_dataset,'testing_files.txt')
    
    # Load files

    print("Loading Files:")

    x_train,y_train = load_data(training_files,sr,file_length,path_dataset,word_to_label)
    x_val,y_val = load_data(validation_files,sr,file_length,path_dataset,word_to_label)
    x_test,y_test = load_data(testing_files,sr,file_length,path_dataset,word_to_label)

    # Load backgrounds separately split and append into partitions

    backgrounds = partition_directory(path_dataset,'_background_noise_',sr,file_length)

    x_train,y_train = append_examples(x_train,y_train,backgrounds[:300],11)
    x_val,y_val = append_examples(x_val,y_val,backgrounds[300:320],11)
    x_test,y_test = append_examples(x_test,y_test,backgrounds[320:],11)

    # Show status
    print("Files and backgrounds loaded:")
    print("X_train shape: ",x_train.shape)
    print("y_train shape: ",y_train.shape)
    print("X_val shape: ",x_val.shape)
    print("y_val shape: ",y_val.shape)
    print("X_test shape: ",x_test.shape)
    print("y_test shape: ",y_test.shape)

    x_train = normalize_waveforms(x_train)
    x_val = normalize_waveforms(x_val)
    x_test = normalize_waveforms(x_test)

    N_train, _ = x_train.shape
    N_val, _ = x_val.shape
    N_test, _ = x_test.shape

    n_classes = len(np.unique(y_train))
    
    y_train_oh = make_oh(y_train)
    y_val_oh = make_oh(y_val)
    y_test_oh = make_oh(y_test)

    # Create Model

    input_shape = (file_length,)
    lr = 0.001

    crnn1D = CRNN1_1D(input_shape, n_classes)
    print("Model Created:")
    crnn1D.summary()

    # Compile Model
    crnn1D.compile(optimizer=Adam(lr),loss='categorical_crossentropy',metrics = ['accuracy'])

    crnn1D.fit(x_train,y_train_oh,
           batch_size=256, epochs = 10,
           validation_data=[x_val,y_val_oh])

main('/home/edoardobucheli/TFSpeechCommands/train/audio')

