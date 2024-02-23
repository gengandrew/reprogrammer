import os
import torch
import numpy as np
import torchvision.transforms as transforms
from utils.label_map import imagenet_classnames
from imagenetv2_pytorch import ImageNetV2Dataset


class ImageNetV2(ImageNetV2Dataset):
    def __init__(self, root='/nobackup-fast/ageng/datasets/ImageNetV2/', transform=None):
        super().__init__(variant="matched-frequency", transform=transform, location=root)
        self.classnames = imagenet_classnames
        self.transform = transform
