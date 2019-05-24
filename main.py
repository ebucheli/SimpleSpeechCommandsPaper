import matplotlib.pyplot as plt
import numpy as np

from ProcessAudio import normalize_waveforms
from Utilities import make_oh, load_dataset, generate_partition
from Utilities import check_combination, choose_network,choose_problem
from Utilities import make_transformation

from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report

import sys
import click

@click.command()
@click.argument('path')
@click.option('--problem', default = 0, help ='Version of the problem:\n\t0:10 words\n\t1:20 words\n\t2:Left/Right')
@click.option('--transformation',default = 0,
              help = 'The transformation to apply:\n0:Waveform\n1:Spectrogram \n2:Mel Spectrogram\n3:MFCC')
@click.option('--mels',default = 40,help = 'Frequency resolution for Mel Spectrogram and MFCC')
@click.option('--network',default = 1,
              help = 'The network to use:\n0:CNN1D\n1:CRNN 1D\n2:AttRNN1D\n3:FCNN\n4:Malley\n5:CNN_TRAD_FPOOL3\n6:CNN_ONE_FSTRIDE4\n7:CRNN 2D V1\n8:CRNN 2D V2\n9:attRNN2D')
@click.option('--train/--no_train',default = True,help = 'Train the model or use pretrained weights, specifiy file using --weights_file')
@click.option('--weights_file', default = '',help = 'Specify the names of the weights to load, automatically saved to /trained_weights/ directory.')
@click.option('--epochs',default = 10, help = 'How many epoch to train the model for')
@click.option('--save_w', is_flag = True ,help = 'Use flag to save the weights of the model.')
@click.option('--outfile', default = '', help = 'If weights are saved, the name of the output file')

def main(path, problem, transformation,mels, network, train, weights_file, epochs, save_w, outfile):

    # Check that the representation and network match

    check_combination(transformation,network)

    #Load dicts with commands and labels

    word_to_label,label_to_word = choose_problem(problem)

    # Set Sample Rate and file length
    sr = 16000
    file_length = 16000
    weights_dir = 'trained_weights/'

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
    print('\n')

    x_train = normalize_waveforms(x_train)
    x_val = normalize_waveforms(x_val)
    x_test = normalize_waveforms(x_test)

    N_train, _ = x_train.shape
    N_val, _ = x_val.shape
    N_test, _ = x_test.shape

    x_train_2,x_val_2,x_test_2,input_shape = make_transformation(transformation,sr,mels,file_length,x_train,x_val,x_test)

    print('\nDone! Input Shape: {}'.format(input_shape))

    n_classes = len(np.unique(y_train))
    print('Number of classes: {}\n'.format(n_classes))

    y_train_oh = make_oh(y_train)
    y_val_oh = make_oh(y_val)
    y_test_oh = make_oh(y_test)

    target_names = []

    for i in np.unique(y_test):
        target_names.append(label_to_word[i])

    # Create Model

    lr = 0.001
    model = choose_network(network,input_shape,n_classes)
    print('Model Created:')
    model.summary()
    print('\n')



    if train:
        # Compile Model
        model.compile(optimizer=Adam(lr),loss='categorical_crossentropy',
                      metrics = ['accuracy'])

        model.fit(x_train_2,y_train_oh,
               batch_size=256, epochs = epochs,
               validation_data=[x_val_2,y_val_oh])

    else:
        print('Loading pre-trained weights:\n')
        model.load_weights(weights_dir+weights_file)

    logits = model.predict(x_test_2)

    y_preds = np.argmax(logits, axis = -1)

    accuracy = np.mean(y_preds==y_test)

    conf_matrix = confusion_matrix(y_test,y_preds)
    report = classification_report(y_test,y_preds,target_names = target_names)
    print('\nAccuracy: {:.2f}%'.format(accuracy*100))
    print('\nConfusion Matrix:\n')
    print(conf_matrix)
    print('\nClassification Report\n')
    print(report)


    if save_w:
        model.save_weights(weights_dir+outfile)
        print('Weights succesfully exported to {}\n'.format(weights_dir+outfile))

if __name__ == '__main__':
    main()
