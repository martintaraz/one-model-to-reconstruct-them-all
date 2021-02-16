import numpy

from pytorch_training.data.utils import default_loader
from torch.utils import data
from typing import Callable, Dict
import os

class DemoDatasetFolder(data.Dataset):

    def __init__(self, folder: str, root: str = None, transforms: Callable = None, loader: Callable = default_loader):
        self.transforms = transforms
        self.loader = loader
        self.image_paths = os.listdir(folder)
        self.folder = folder
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, numpy.ndarray]:
        image = self.loader(os.path.join(self.folder, self.image_paths[index]))

        if self.transforms is not None:
            image = self.transforms(image)

        return {
            'input_image': image,
            'output_image': image
        }

