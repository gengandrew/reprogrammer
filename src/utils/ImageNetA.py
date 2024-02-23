import os
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils.label_map import imagenet_a_indices, imagenet_a_classnames


class ImageNetA(datasets.ImageFolder):
    def __init__(self, root='./datasets/ImageNet-A', transform=None):
        super().__init__(root, transform)
        
        self.classnames = imagenet_a_classnames
        self.transform = transform