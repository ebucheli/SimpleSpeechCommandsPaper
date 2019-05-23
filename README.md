# Simple Speech Commands Detection
---

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

There are also four basic input representations for audio.

1. Waveforms
1. Power Spectrograms
1. Mel Spectrograms
1. Mel Frequency Cepstral Coefficients (MFCC)

Using `main.py` you can create a model with a combination of these two aspects. However, architectures made for waveforms are only compatible with waveforms.

## Usage

You can execute using `python main.py [OPTIONS]` from the command line.

You can use the `--help` flag to learn about the usage. Below is a description of the available options.

* `--path [the path to the dataset in your system]`: use this flag to point where the data set is in your system. It expects the directory with the subfolders for every word i.e. `/yes/`,`/no/`, etc.
* `--problem [INTEGER]`: Select the version of the problem, here you can choose between classifying 0:10 words (plus unknown and silence), 1:20 words (plus unknown and silence) and only 2:Left/Right (plus unknown and silence).
* `--transformation [INTEGER]`: Select the input representation, 0: Waveform, 1: Power Spectrogram 2: Mel Spectrogram 3: MFCC.
* `--mels [INTEGER]`: Choose the Frequency resolution for Mel Spectrograms and MFCC, one of 40, 80 and 120.
* `--network [INTEGER]`: Choose the Architecture, use `--help` for a breakdown.
* `--train/--no_train`:  If you wish to use pre-trained weights, use `--no_train`. If so please specify the file using `--weights_file`.
* `--weights_file [TEXT]`: Name of the file with the pre-trained weights, the package assumes it is in `trained_weights`. Use `--no_train`.
* `--epochs [INTEGER]`: Specify the number of epochs.
* `--save_w`: Use this flag to save the weights after the model has been trained. Please specify the name of the file using `--outfile`
* `--outfile [TEXT]`: Specify the name of the output file with the weights.
