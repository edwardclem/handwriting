# handwriting
Simple CRNN handwriting recognition. 


## Overview

The goal of this repo is to implement a simple CRNN for line-level handwriting recognition method. The implementation in `hw/models/crnn.py` contains a `LightningModule` that uses a ResNet image feature encoder (implemented in `hw/models/cnn_encoder.py`) and a few LSTM layers to perform the final sequence prediction. The model is trained using the `CTCLoss` implemented in `torch.nn` and beam-search inference is performed using either greedy decoding or the beam search decoders from `torchvision`. 


## Installation

clone and:

    pip install .

## Dataset

The IAM-Line dataset is pulled from huggingface datasets, specifically [this one](https://huggingface.co/datasets/Teklia/IAM-line). For some reason, the original IAM handwriting recognition dataset [website](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) never sent me an email - maybe I'm not cool. This has the added bonus of being easily pulled as long as huggingface datasets installed. The dataset is essentially a set of `(image, text_line)` tuples. I put together a `LightningDataModule` that basically just wraps the train, test, and val dataloaders with some `torchvision` transforms and collates. Not exactly rocket surgery. 

## Training

An example training script can be found in `train.py`, which uses the Lighting `Trainer` abstraction and some callbacks to fit a model from scratch. I can get a character error rate of around 0.063 on the IAM validation partition (with beam search decoding!) and the provided set of data augmentations in about ~2 hours (100 epochs) on an RTX 2070. 

## Notes

- Putting this together was overall not too tricky - but I spent forever figuring out that the order of the `AdaptiveAveragePool` dimensions should be _reversed_ from how it was used in the initial implementation I started working with. Oddly enough, the usual learning-not-working debug trick of trying to overfit on a small dataset still worked. I find it helpful to think through the shapes of the CNN feature maps when debugging anything CNN/ResNet related
- I didn't find the shortcut trick from [this paper](https://arxiv.org/abs/2404.11339) terribly helpful, but it's a fun idea. Currently it's implemented as the default `CRNN` option, but it can be easily disabled.
- I didn't end up putting a language model in here - that would require using a different decoder as well.   
- I found that beam search only improves aggregate CER by a percentage point or so, but, anecdotally, the results are much better. Makes me think there's something misleading about CER? Probably a WER vs CER sort of thing? This might be worth poking at further. 

## Notebooks

I've provided a few example notebooks - `examples/dataset.ipynb` has some poking around in the IAM dataset, and `examples/inference.ipynb` has some inference examples and test-time stats. The provided artifact was trained without the training shortcut. 

## Citations

This repo was inspired by [this one] (https://github.com/jc639/pytorch-handwritingCTC/tree/master) from @jc639, and implemements some tricks from https://arxiv.org/abs/2404.11339 - GitHub [here](https://github.com/georgeretsi/HTR-best-practices/tree/main). This blog post from [distil.pub](https://distill.pub/2017/ctc/) was very helpful in understanding the CTC loss function. I borrowed the implementation of beam search in `hw/models/beam.py` from [this](https://gist.github.com/awni/56369a90d03953e370f3964c826ed4b0) github gist, although it won't surprise you to learn that the CUDA beam search decoder in torchaudio is much faster. 
