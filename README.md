# Simple Speech Commands Detection

This repo presents various architectures to solve the Keyword Detection Problem using Deep Learning and the [Simple Speech Commands Detection](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz) dataset from TensorFlow.
This code was generated as part of a research project to be published at COMIA 2019 in Mexico, currently only available in spanish.
The name of the paper is "Deteccion de Comandos de Voz con Modelos Compactos de Aprendizaje Profundo" (A Review of Small-Footprint Speech Commands Detection Using Deep Learning)


## Structure

This research was a comparative study of audio representations and artificial neural networks.

You can find four basic types of Neural Network Architectures in this repository.

1. MLP (in `FFNetworks.py`)
1. 1D Convnets (in `CNNetworks1D.py`)
1. 2D Convnets (in `CNNetworks2D.py`)
1. CRNN (in `RNNetworks.py`)

There are also four basic input representations for audio. 2, 3 and 4 are all based in the Fourier Transform and are all time-frequency representations.

1. Waveforms
1. Power Spectrograms
1. Mel Spectrograms
1. Mel Frequency Cepstral Coefficients (MFCC)

Using `main.py` you can create a model with a combination of these two aspects. However, architectures made for waveforms are only compatible with waveforms.

## Usage

You can execute using `python main.py [OPTIONS] PATH` from the command line.

You need to specify `PATH` to point where the data set is in your system. It expects the directory with the subfolders for every word i.e. `/yes`,`/no`, etc.

You can use the `--help` flag to learn about the usage. Below is a description of the available options.

* `--problem [INTEGER]`: Select the version of the problem, here you can choose between classifying 10 words (plus unknown and silence) (0), 20 words (plus unknown and silence) (1) and 2Words (Left/Right) (plus unknown and silence) (2). If you choose 10 Words you need v0.01 of the dataset.
* `--transformation [INTEGER]`: Select the input representation; Waveform (0), Power Spectrogram (1), Mel Spectrogram (2), or MFCC (3). If no transformation is specified, Waveforms will be used for Networks 0, 1 and 2 and MS-40 for the rest.
* `--mels [INTEGER]`: Choose the Frequency resolution for Mel Spectrograms and MFCC (either 40, 80 or 120).
* `--network [INTEGER]`: Choose the Architecture, check next section or use `--help` for a breakdown.
* `--train/--no_train`:  If you wish to use pre-trained weights, use `--no_train`. If so please specify the file using `--weights_file`.
* `--weights_file [TEXT]`: Name of the file with the pre-trained weights, the package assumes it is in `trained_weights`. Use `--no_train`.
* `--epochs [INTEGER]`: Specify the number of epochs.
* `--save_w`: Use this flag to save the weights after the model has been trained. Please specify the name of the file using `--outfile`
* `--outfile [TEXT]`: Specify the name of the output file with the weights.

## Networks

We have provided 10 different Artificial Neural Network architectures. The first three are meant for waveforms and thus can't be used with time-frequency representations, the inverse is true for the final seven networks.

0. CNN 1D: Taken from [here](https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data).
1. CRNN 1D: This network was inspired by [this paper](https://arxiv.org/abs/1703.05390) but applied to waveforms.
2. attRNN 1D: Same as before but with [this paper](https://arxiv.org/abs/1808.08929).
3. FCNN: Baseline model described [here](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43969.pdf)
4. O'Malley: This model was proposed by the second place winner of the TensorFlow Speech Recognition Challenge in Kaggle, Thomas O'Malley. He explains it [here](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/47715).
5. CNN_TRAD_FPOOL3: From the same paper as FCNN.
6. CNN_ONE_FSTRIDE4: From the same paper as FCNN and CNN_TRAD_FPOOL3.
7. CRNN1 2D: This is on of the networks from the same paper as the one referenced in CRNN 1D.
8. CRNN2 2D: Same as CRNN1 2D.
9. attRNN 2D: This network was the one proposed in the paper referenced in attRNN 1D and you can also find it [here](https://github.com/douglas125/SpeechCmdRecognition).

## Examples

There are defaults for all the options, if you run, `python main.py` the model will run using waveforms and the architecture CNN1D on the 10 Words problem. The model will be trained but the weights will not be saved.

If you want to use a Mel Spectrogram with a frequency resolution of 80 on the 20 word problem and the CRNN1-2D architecture and save the weights you can run,

`python main.py --problem 1 --transformation 2 --mels 80 --network 7 --save_w --outfile "example_weights.h5"`

You can also use pre-trained weights either created by you or from some of our previously generated one in `trained_weights`. The package assumes the file is in said directory;

`python main.py --no_train --weights_file 'WF_CNN1D_10Words.h5'`

## Requirements

The models were created using Python 3.6 with Keras using the TensorFlow backend. You will also need the [Librosa](https://librosa.github.io/librosa/) package. Other dependencies includes the [tqdm](https://tqdm.github.io/) package.
