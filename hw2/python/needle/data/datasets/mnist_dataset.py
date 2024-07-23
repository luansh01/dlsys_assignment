from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(label_filename) as f:
            magic_number = int.from_bytes(f.read(4), 'big')
            if magic_number == 2049:
                num_items = int.from_bytes(f.read(4), 'big')
            labels_uint8 = np.frombuffer(f.read(), dtype = np.uint8)

        with gzip.open(image_filename) as f:
            magic_number = int.from_bytes(f.read(4), 'big')
            if magic_number == 2051:
                num_items = int.from_bytes(f.read(4), 'big')
                num_rows = int.from_bytes(f.read(4), 'big')
                num_cols = int.from_bytes(f.read(4), 'big')
                images_uint8 = np.frombuffer(f.read(),dtype = np.uint8)
                images_uint8 = images_uint8.reshape(num_items,num_rows,num_cols)
                images_float32 = (images_uint8/255.0) .astype(np.float32)

        self.images_float32 = images_float32
        self.label_int8 = labels_uint8
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        #print(self.images_float32.shape)
        image = self.images_float32[index]
        label = self.label_int8[index]
        if self.transforms != None:
            for tran in self.transforms:
                tran(image)
        return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images_float32.shape[0]
        raise NotImplementedError()
        ### END YOUR SOLUTION