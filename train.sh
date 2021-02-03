#!/bin/bash
python train_code_finder.py configs/autoencoder.yaml stylegan-256px-new.model  --images dataset/train.json  -s 1 --neural_rendering
