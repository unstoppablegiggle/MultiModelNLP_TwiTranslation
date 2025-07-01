# Twi Translation System

## Introduction
This project is an exploration of NLP models such as OpenAIs, Whisper, and
Meta's No Language Left Behind model (NLLB), it's primary goal was to develop
a simple straightforward method for an elderly twi speaker to easily communicate
with english speakers with relative efficiency using a simple android app connected
to the models.

The project was a success but is limited by the quality and capabilities of the models
available, notably, english to twi translation is reasonably accurate however the models 
struggle to handle twi to english translation. This is due to Twi being a low resource language
as noted in the [NLLB paper](https://arxiv.org/pdf/2207.04672).

## Approach

The pipeline is designed to be plug and play such that new models can be dropped in
as needed. Technically this system will work with *any language*.

The system:
- Spin up locally available server with Flask
- load models: Whisper, Whisper trained on Twi, NLLB
- Audio file is sent from custom app on phone
- The Whisper models handle STT and TTS, NLLB handles translation
- Audio file and text is returned to app for display and playback in .JSON format.

## Setup

- Set up a conda environment using the provided .yml file
- Run the server (note the models do need a bit of headspace from the GPU)
- downlaod and run the app (Dev mode on for Android)

It should be noted the app sets off google's malware detection, I promise with all my heart it's **not** a virus,
but as with anything on the internet please use your best judgement.
