import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio
import librosa
import librosa.display

from SimpleSpeechCommands import get_word_dict, read_list, load_data
from SimpleSpeechCommands import append_examples,partition_directory
from SimpleSpeechCommands import get_dataset_partition, reduce_examples
from ProcessAudio import normalize_waveforms
from Utilities import make_oh

from tensorflow.keras.optimizers import Adam

import sys

import click

@click.command()
@click.option('--path',default = '/home/edoardobucheli/TFSpeechCommands/train/audio',help='Path to the dataset')
@click.option('--problem', default = 0, help ='Version of the problem:\n\t0:10 words\n\t1:20 words\n\t2:Left/Right')
@click.option('--transformation',default = 0,
              help = 'The transformation to apply:\n0:Waveform\n1:Spectrogram \n2:Mel Spectrogram\n3:MFCC')
@click.option('--mels',default = 40,help = 'Frequency resolution for Mel Spectrogram and MFCC')
@click.option('--network',default = 1,
              help = 'The network to use:\n0:CNN1D\n1:CRNN 1D\n2:AttRNN1D\n3:FCNN\n4:Malley\n5:CNN_TRAD_FPOOL3\n6:CNN_ONE_FSTRIDE4\n7:CRNN 2D V1\n8:CRNN 2D V2\n9:attRNN2D')
@click.option('--train',default = True,help = '(NOT IMPLEMENTED)Train the model or use pretrained weights')
@click.option('--epochs',default = 10, help = 'How many epoch to train the model for')

def main(path, problem, transformation,mels, network, train, epochs):


    # Check that the representation and network match

    check_combination(transformation,network)

    #Load dicts with commands and labels

    word_to_label,label_to_word = choose_problem(problem)

    # Set Sample Rate and file length
    sr = 16000
    file_length = 16000

    # Get partition lists

    training_files, validation_files, testing_files = generate_partition(path,problem, word_to_label)

    # Load files


    x_train,y_train,x_val,y_val,x_test,y_test = load_dataset(training_files,validation_files,testing_files,
                                                             sr,file_length,path,word_to_label,problem)


    # Show status
    print("\nFiles and backgrounds loaded!\n")
    print("\tTraining Examples: ",len(x_train))
    print("\tValidation Examples: ",len(x_val))
    print("\tTesting Examples: ",len(x_test),end = '\n\n')

    print('Label Distribution:')
    for i in range(len(np.unique(y_train))):
        print("{}: {}".format(label_to_word[i],np.sum(y_train==i)))

    x_train = normalize_waveforms(x_train)
    x_val = normalize_waveforms(x_val)
    x_test = normalize_waveforms(x_test)

    N_train, _ = x_train.shape
    N_val, _ = x_val.shape
    N_test, _ = x_test.shape

    x_train_2,x_val_2,x_test_2,input_shape = make_transformation(transformation,sr,mels,file_length,x_train,x_val,x_test)

    print('\nDone! Input Shape: {}\n'.format(input_shape))

    n_classes = len(np.unique(y_train))

    print(n_classes)

    y_train_oh = make_oh(y_train)
    y_val_oh = make_oh(y_val)
    y_test_oh = make_oh(y_test)

    # Create Model

    lr = 0.001

    model = choose_network(network,input_shape,n_classes)

    print('Model Created:')
    model.summary()
    print('\n\n')

    # Compile Model
    model.compile(optimizer=Adam(lr),loss='categorical_crossentropy',metrics = ['accuracy'])

    model.fit(x_train_2,y_train_oh,
           batch_size=256, epochs = epochs,
           validation_data=[x_val_2,y_val_oh])

def load_dataset(training_files,validation_files,testing_files,sr,file_length,path,word_to_label,problem):

    print("\nLoading Files:\n")

    x_train,y_train = load_data(training_files,sr,file_length,path,word_to_label)
    x_val,y_val = load_data(validation_files,sr,file_length,path,word_to_label)
    x_test,y_test = load_data(testing_files,sr,file_length,path,word_to_label)

    # Load backgrounds separately split and append into partitions

    backgrounds = partition_directory(path,'_background_noise_',sr,file_length)

    if problem == 0:
        bkg_label = 11
    elif problem == 1:
        bkg_label = 21
    elif problem == 2:
        bkg_label = 3

    x_train,y_train = append_examples(x_train,y_train,backgrounds[:300],bkg_label)
    x_val,y_val = append_examples(x_val,y_val,backgrounds[300:320],bkg_label)
    x_test,y_test = append_examples(x_test,y_test,backgrounds[320:],bkg_label)

    return x_train,y_train,x_val,y_val,x_test,y_test

def generate_partition(path, problem, word_to_label):
    training_files, validation_files, testing_files = get_dataset_partition(path,10,10)

    if problem == 0:
        unk_label = 10
        unk_keep = 0.2
    elif problem == 1:
        unk_label = 20
        unk_keep = 0.2
    elif problem == 2:
        unk_label = 2
        unk_keep = 0.1

    training_files = reduce_examples(training_files,unk_label,unk_keep,word_to_label)
    validation_files = reduce_examples(validation_files,unk_label,unk_keep,word_to_label)
    testing_files = reduce_examples(testing_files,unk_label,unk_keep,word_to_label)


    """
    You can also load previously generated lists that you can generate using the functions in SimpleSpeechCommands.py

    training_files = read_list(path,'training_files.txt')
    validation_files = read_list(path,'validation_files.txt')
    testing_files = read_list(path,'testing_files.txt')
    """

    return training_files, validation_files, testing_files

def check_combination(transformation,network):

    if network < 0 or network > 9:
        print('Invalid network selection')
        sys.exit()

    if transformation == 0:
        if network != 0 and network != 1 and network != 2:
            print('The selected representation and network do not match')
            sys.exit()
    elif transformation > 0 and transformation <= 3:
        if network == 0 or network == 1 or network == 2:
            print('The selected representation and network do not match')
            sys.exit()
    elif transformation < 0 or transformation > 3:
        print('Invalid Selection for transformation')
        sys.exit()

def choose_network(network,input_shape,n_classes):

    freq_res = input_shape[0]

    if network == 0:
        # CNN 1D
        from CNNetworks1D import conv1d_v1
        model = conv1d_v1(input_shape,n_classes)

    elif network == 1:
        # CRNN 1D
        from RNNetworks import CRNN1_1D
        model = CRNN1_1D(input_shape,n_classes)

    elif network == 2:
        # attRNN 1D
        from RNNetworks import AttRNNSpeechModelWave
        model = AttRNNSpeechModelWave(input_shape,n_classes)

    elif network == 3:
        # Fully Connected NN
        from FFNetworks import DNN_3HL
        model = DNN_3HL(input_shape,n_classes)

    elif network == 4:
        # Malley CNN
        if freq_res == 40:
            from CNNetworks2D import malley_cnn_40
            model = malley_cnn_40(input_shape,n_classes)
        elif freq_res == 80:
            from CNNetworks2D import malley_cnn_80
            model = malley_cnn_80(input_shape,n_classes)
        elif freq_res >= 120:
            from CNNetworks2D import malley_cnn_120
            model = malley_cnn_120(input_shape,n_classes)

    elif network == 5:
        # CNN TRAD FPOOL 3
        if freq_res == 40:
            from CNNetworks2D import cnn_trad_fpool3_40
            model = cnn_trad_fpool3_40(input_shape,n_classes)
        elif freq_res >= 120  or freq_res == 80:
            from CNNetworks2D import cnn_trad_fpool3_120
            model = cnn_trad_fpool3_120(input_shape,n_classes)

    elif network == 6:
        # CNN ONE FSTRIDE
        if freq_res == 40:
            from CNNetworks2D import cnn_one_fstride4_40
            model = cnn_one_fstride4_40(input_shape,n_classes)
        elif freq_res >= 120  or freq_res == 80:
            from CNNetworks2D import cnn_one_fstride4_120
            model = cnn_one_fstride4_120(input_shape,n_classes)

    elif network == 7:
        # CRNN 2D V1
        from RNNetworks import CRNN_v1
        model = CRNN_v1(input_shape,n_classes)

    elif network == 8:
        # CRNN 2D V2
        from RNNetworks import CRNN_v2
        model = CRNN_v2(input_shape,n_classes)

    elif network == 9:
        # attRNN 2D
        from RNNetworks import AttRNNSpeechModel
        model = AttRNNSpeechModel(input_shape,n_classes)
    else:
        print("Please choose a valid Neural Network, to learn more use --help")

    return model

def choose_problem(problem):

    if problem == 0:
        from SimpleSpeechCommands import get_word_dict
        word_to_label,label_to_word = get_word_dict()
    elif problem == 1:
        from SimpleSpeechCommands import get_word_dict_v2
        word_to_label,label_to_word = get_word_dict_v2()
    elif problem == 2:
        from SimpleSpeechCommands import get_word_dict_2words
        word_to_label,label_to_word = get_word_dict_2words()
    else:
        print('Please select a problem between 0-2 for more info use --help flag.')
        sys.exit()
    return word_to_label,label_to_word

def make_transformation(transformation, sr, mels, file_length, x_train, x_val, x_test):

    if transformation == 0:
        x_train_2 = x_train
        x_val_2 = x_val
        x_test_2 = x_test

        input_shape = (file_length,)

    elif transformation == 1:
        print('Power Spectragram Selected, generating representation:\n')
        from ProcessAudio import power_spect_set

        n_fft = 512
        hop_length = 512

        freq_res = 257
        frames = 32

        x_train_2 = power_spect_set(x_train,sr,n_fft,hop_length)
        x_val_2 = power_spect_set(x_val,sr,n_fft,hop_length)
        x_test_2 = power_spect_set(x_test,sr,n_fft,hop_length)

        input_shape = (freq_res,frames)

    elif transformation == 2:
        print('Mel Spectragram Selected, generating representation:\n')
        from ProcessAudio import mel_spec_set

        n_mels = mels
        hop_length = 512
        frames = 32

        x_train_2 = mel_spec_set(x_train,sr,n_mels,hop_length)
        x_val_2 = mel_spec_set(x_val,sr,n_mels,hop_length)
        x_test_2 = mel_spec_set(x_test,sr,n_mels,hop_length)

        input_shape = (n_mels,frames)

    elif transformation == 3:
        print('MFCC Selected, generating representation:\n')
        from ProcessAudio import mfcc_set

        n_mfcc = mels
        hop_length = 512
        frames = 32

        x_train_2 = mfcc_set(x_train,sr,n_mfcc,hop_length)
        x_val_2 = mfcc_set(x_val,sr,n_mfcc,hop_length)
        x_test_2 = mfcc_set(x_test,sr,n_mfcc,hop_length)

        input_shape = (n_mfcc,frames)

    return x_train_2,x_val_2,x_test_2,input_shape

if __name__ == '__main__':
    main()
