import os
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils.label_map import imagenet_r_indices, imagenet_r_classnames


class ImageNetR(datasets.ImageFolder):
    def __init__(self, root='./datasets/ImageNet-R', transform=None):
        super().__init__(root, transform)
        
        self.classnames = imagenet_r_classnames
        self.transform = transform