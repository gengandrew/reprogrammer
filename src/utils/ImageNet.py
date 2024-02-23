import os
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils.label_map import imagenet_classnames, imagenet100_classnames


class ImageNet(datasets.ImageFolder):
    def __init__(self, root='/nobackup-fast/ageng/datasets/ImageNet-1k/', train=True, transform=None):
        self.classnames = imagenet_classnames
        self.transform = transform

        if train:
            root = root + 'train'
        else:
            root = root + 'val'

        super().__init__(root, transform)
