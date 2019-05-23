# Simple Speech Commands Detection
---

This repo presents various architectures to solve the Keyword Detection Problem using Deep Learning and the Simple Speech Commands Detection dataset from TensorFlow.
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
1. Mel Frequency Cepstral Coefficients

Using `main.py` you can create a model with a combination of these two aspects. However, architectures made for waveforms are only compatible with waveforms.

## Usage

You can execute `python main.py [options]` from the command line, using `python main.py --help` the following appears:

Usage: main.py [OPTIONS]

Options:
  --path TEXT               Path to the dataset
  --problem INTEGER         Version of the problem:
                            0:10 words
                            1:20 words
                            2:Left/Right
  --transformation INTEGER  The transformation to apply:
                            0:Waveform
                            1:Spectrogram
                            2:Mel Spectrogram
                            3:MFCC
  --mels INTEGER            Frequency resolution for Mel Spectrogram and MFCC
  --network INTEGER         The network to use:
                            0:CNN1D
                            1:CRNN 1D
                            2:AttRNN1D
                            3:FCNN
                            4:Malley
                            5:CNN_TRAD_FPOOL3
                            6:CNN_ONE_FSTRIDE4
                            7:CRNN 2D V1
                            8:CRNN 2D V2
                            9:attRNN2D
  --train / --no_train      Train the model or use pretrained weights,
                            specifiy file using --weights_file
  --weights_file TEXT       Specify the names of the weights to load,
                            automatically saved to /trained_weights/
                            directory.
  --epochs INTEGER          How many epoch to train the model for
  --save_w                  Use flag to save the weights of the model.
  --outfile TEXT            If weights are saved, the name of the outpu
