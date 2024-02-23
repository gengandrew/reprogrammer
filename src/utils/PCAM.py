import h5py
from PIL import Image
from torch.utils.data import Dataset


class PCAM(Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = path
        self.dataset_x = None
        self.dataset_y = None
        self.transform = transform

        if train:
            self.file_name = '/camelyonpatch_level_2_split_train'
        else:
            self.file_name = '/camelyonpatch_level_2_split_test'

        # Loading images part of HDF5
        with h5py.File(self.path + self.file_name + '_x.h5', 'r') as filex:
            self.dataset_x_len = len(filex['x'])

        # Loading label part of HDF5
        with h5py.File(self.path + self.file_name + '_y.h5', 'r') as filey:
            self.dataset_y_len = len(filey['y'])

    def __len__(self):
        assert self.dataset_x_len == self.dataset_y_len
        return self.dataset_x_len

    def __getitem__(self, index):
        imgs_path = self.path + self.file_name + '_x.h5'
        labels_path = self.path + self.file_name + '_y.h5'

        if self.dataset_x is None:
            self.dataset_x = h5py.File(imgs_path, 'r')['x']
        if self.dataset_y is None:
            self.dataset_y = h5py.File(labels_path, 'r')['y']

        # Get one pair of X, Y and return them, transform if needed
        image = Image.fromarray(self.dataset_x[index]).convert("RGB")
        label = self.dataset_y[index, 0, 0, 0]

        if self.transform:
            image = self.transform(image)

        return (image, label)