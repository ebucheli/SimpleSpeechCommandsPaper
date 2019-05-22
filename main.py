import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio
import librosa
import librosa.display

from SimpleSpeechCommands import get_word_dict, read_list, load_data
from SimpleSpeechCommands import append_examples,partition_directory
from ProcessAudio import normalize_waveforms
from Utilities import make_oh

from tensorflow.keras.optimizers import Adam

import sys

import click

@click.command()
@click.option('--path',default = '/home/edoardobucheli/TFSpeechCommands/train/audio',help='Path to the dataset')
@click.option('--problem', default = 0, help ='Version of the problem:\n\t0 = 10 words\n\t1 = 20 words\n\t2 = Left/Right')
@click.option('--transformation',default = 0,
              help = 'The transformation to apply:\n 0 = Waveform \n 1 = Spectrogram \n 2 = Mel Spectrogram \n 3 = MFCC')
@click.option('--network',default = 0,help = 'The network to use')
@click.option('--train',default = True,help = 'Train the model or use pretrained weights')


def main(path, problem, transformation, network, train):

    #Load dicts with commands and labels

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

    # Set Sample Rate and file length
    sr = 16000
    file_length = 16000

    # Load filenames from previously generated lists

    training_files = read_list(path,'training_files.txt')
    validation_files = read_list(path,'validation_files.txt')
    testing_files = read_list(path,'testing_files.txt')

    # Load files

    print("\nLoading Files:\n")

    x_train,y_train = load_data(training_files,sr,file_length,path,word_to_label)
    x_val,y_val = load_data(validation_files,sr,file_length,path,word_to_label)
    x_test,y_test = load_data(testing_files,sr,file_length,path,word_to_label)

    # Load backgrounds separately split and append into partitions

    backgrounds = partition_directory(path,'_background_noise_',sr,file_length)

    x_train,y_train = append_examples(x_train,y_train,backgrounds[:300],11)
    x_val,y_val = append_examples(x_val,y_val,backgrounds[300:320],11)
    x_test,y_test = append_examples(x_test,y_test,backgrounds[320:],11)

    # Show status
    print("\nFiles and backgrounds loaded!\n")
    print("\tTraining Examples: ",len(x_train))
    print("\tValidation Examples: ",len(x_val))
    print("\tTesting Examples: ",len(x_test),end = '\n\n\n')

    x_train = normalize_waveforms(x_train)
    x_val = normalize_waveforms(x_val)
    x_test = normalize_waveforms(x_test)

    N_train, _ = x_train.shape
    N_val, _ = x_val.shape
    N_test, _ = x_test.shape

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

        print('\nDone! Input Shape: ({},{})\n'.format(freq_res,frames))

    elif transformation == 2:
        print('Mel Spectragram Selected, generating representation:\n')
        from ProcessAudio import mel_spec_set

        n_mels = 40
        hop_length = 512
        frames = 32

        x_train_2 = mel_spec_set(x_train,sr,n_mels,hop_length)
        x_val_2 = mel_spec_set(x_val,sr,n_mels,hop_length)
        x_test_2 = mel_spec_set(x_test,sr,n_mels,hop_length)

        input_shape = (n_mels,frames)
        print('\nDone! Input Shape: ({},{})\n'.format(n_mels,frames))

    elif transformation == 3:
        print('MFCC Selected, generating representation:\n')
        from ProcessAudio import mfcc_set

        n_mfcc = 40
        hop_length = 512
        frames = 32

        x_train_2 = mfcc_set(x_train,sr,n_mfcc,hop_length)
        x_val_2 = mfcc_set(x_val,sr,n_mfcc,hop_length)
        x_test_2 = mfcc_set(x_test,sr,n_mfcc,hop_length)

        input_shape = (n_mfcc,frames)
        print('\nDone! Input Shape: ({},{})\n'.format(n_mfcc,frames))



    n_classes = len(np.unique(y_train))

    y_train_oh = make_oh(y_train)
    y_val_oh = make_oh(y_val)
    y_test_oh = make_oh(y_test)

    # Create Model


    lr = 0.001

    if network == 0:
        from RNNetworks import CRNN1_1D
        model = CRNN1_1D(input_shape,n_classes)

    elif network == 1:
        from CNNetworks2D import malley_cnn_40
        model = malley_cnn_40(input_shape,n_classes)

    print('Model Created:')
    model.summary()

    # Compile Model
    model.compile(optimizer=Adam(lr),loss='categorical_crossentropy',metrics = ['accuracy'])

    model.fit(x_train_2,y_train_oh,
           batch_size=256, epochs = 10,
           validation_data=[x_val_2,y_val_oh])


if __name__ == '__main__':
    main()
