import os
import torch
import torchvision
import numpy as np
from torchvision import transforms
from torchvision.datasets import VisionDataset


def convert(x):
    if isinstance(x, np.ndarray):
        return torchvision.transforms.functional.to_pil_image(x)
    
    return x


class CIFAR101(VisionDataset):
    def __init__(self, root='./datasets/cifar10.1', transform=None, target_transform=None):
        if transform is not None:
            transform.transforms.insert(0, convert)
        
        super(CIFAR101, self).__init__(root=None, transform=transform, target_transform=target_transform)

        self.images = np.load(os.path.join(root, 'cifar10.1_v6_data.npy'), allow_pickle=True)
        self.targets = torch.Tensor(np.load(os.path.join(root, 'cifar10.1_v6_labels.npy'), allow_pickle=True)).long()

        assert len(self.images) == len(self.targets)
        self.classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.targets[index]

    def __len__(self):
        return len(self.targets)