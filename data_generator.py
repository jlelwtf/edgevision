import cv2
import numpy as np
import os

from keras.utils.data_utils import Sequence


class DataGenerator(Sequence):
    def __init__(
            self,
            dataset_dicts,
            batch_size=32,
            image_size=(244, 244, 3),
            shuffle=True,
            image_dir=''
    ):
        self.image_dir = image_dir
        self.indexes = np.arange(len(dataset_dicts))
        self.dataset_dicts = dataset_dicts
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def _load_data(self, indexes):
        x = np.zeros([len(indexes)] + list(self.image_size)).astype(np.uint8)
        y = np.zeros((len(indexes), 2))
        for i, index in enumerate(indexes):
            note = self.dataset_dicts[index]
            image = cv2.imread(os.path.join(self.image_dir, note['image_name']))
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            x[i, ] = image
            label = np.zeros((2,))
            label[note['class']] = 1
            y[i, ] = label
        return x, y

    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index+1) * self.batch_size]
        return self._load_data(indexes)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
        pass

