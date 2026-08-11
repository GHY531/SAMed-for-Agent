import os
import numpy as np
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import random
import torch
from einops import repeat

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=3, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long()}
        return sample

class DirectDataset(Dataset):
    """
    From txt which contains absolute path of label npys,
    Reads, filters, and indexes valid slices containing Class 2 (Aorta) and non-zero labels.
    """
    def __init__(self, label_path, transform=None):
        self.transform = transform
        self.index = []

        with open(label_path) as f:
            label_paths = [line.strip() for line in f.readlines() if line.strip()]
        
        for label_path in label_paths:
            if not os.path.exists(label_path):
                continue

            label = np.load(label_path)

            if np.any(label) and (2 in label):
                self.index.append(label_path)

    def get_image_path(self, label_path):
        """
        get image path form label path
        """
        dir_name = label_path
        
        image_dir = dir_name.replace('label', 'image')

        return image_dir
        

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        label_path = self.index[idx]
        image_path = self.get_image_path(label_path)

        image = np.load(image_path)
        label = np.load(label_path)

        image = np.array(image, dtype=np.float32)
        label = np.array(label)

        image = image / 255.0

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        file_stem = os.path.splitext(os.path.basename(label_path))[0]
        sample['case_name'] = file_stem

        return sample