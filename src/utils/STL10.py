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


class STL10(VisionDataset):
    def __init__(self, root='./datasets/stl10', train=True, transform=None, target_transform=None, remove_monkeys=True):
        if transform is not None:
            transform.transforms.insert(0, convert)
        
        super(STL10, self).__init__(root=None, transform=transform, target_transform=target_transform)

        if train:
            self.images = self.read_images(os.path.join(root, 'train_X.bin'))
            self.targets = torch.Tensor(self.read_labels(os.path.join(root, 'train_y.bin'))).long()
        else:
            self.images = self.read_images(os.path.join(root, 'test_X.bin'))
            self.targets = torch.Tensor(self.read_labels(os.path.join(root, 'test_y.bin'))).long()

        if remove_monkeys:
            self.remove_monkeys()
        
        assert len(self.images) == len(self.targets)
        self.classnames = ['airplane','bird','automobile','cat','deer','dog','horse','monkey','ship','truck']

    def __getitem__(self, index):        
        return self.transform(self.images[index]), self.targets[index]

    def __len__(self):
        return len(self.targets)

    def read_images(self, path_to_data):
        with open(path_to_data, 'rb') as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)

            # We force the data into 3x96x96 chunks, since the
            # images are stored in "column-major order", meaning
            # that "the first 96*96 values are the red channel,
            # the next 96*96 are green, and the last are blue."
            # The -1 is since the size of the pictures depends
            # on the input file, and this way numpy determines
            # the size on its own.
            images = np.reshape(everything, (-1, 3, 96, 96))

            # Now transpose the images into a standard image format
            # readable by, for example, matplotlib.imshow
            # You might want to comment this line or reverse the shuffle
            # if you will use a learning algorithm like CNN, since they like
            # their channels separated.
            images = np.transpose(images, (0, 3, 2, 1))
            return images
    
    def read_labels(self, path_to_labels):
        with open(path_to_labels, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8) - 1
            return labels

    def remove_monkeys(self):
        indices = np.where(self.targets == 7)[0]
        mask = torch.ones(self.targets.numel(), dtype=torch.bool)
        mask[indices] = False

        self.images = self.images[mask]
        self.targets = self.targets[mask]