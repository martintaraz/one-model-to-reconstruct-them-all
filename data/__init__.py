import argparse
from typing import Type

from data.autoencoder_dataset import AutoencoderDataset, DenoisingAutoencoderDataset, \
    BlackAndWhiteDenoisingAutoencoderDataset, NeuralRenderingDataset


def get_dataset_class(args: argparse.Namespace) -> Type[AutoencoderDataset]:
    if getattr(args, 'denoising', False):
        dataset_class = DenoisingAutoencoderDataset
    elif getattr(args, 'black_and_white_denoising', False):
        dataset_class = BlackAndWhiteDenoisingAutoencoderDataset
    elif getattr(args, 'neural_rendering', False):
        print("selected Neural Rendering Dataset")
        dataset_class = NeuralRenderingDataset
    else:
        print("fell back to default: AutoencoderDataset")
        dataset_class = AutoencoderDataset
    return dataset_class
