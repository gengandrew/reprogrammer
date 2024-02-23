import os
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils.label_map import imagenet_classnames


class ImageNetS(datasets.ImageFolder):
    def __init__(self, root='./datasets/ImageNet-Sketch', transform=None):
        super().__init__(root, transform)
        
        self.classnames = imagenet_classnames
        self.transform = transform
