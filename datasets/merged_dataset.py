import os
import glob
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
    image = ndimage.rotate(image, angle, order=0, reshape=False)
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
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long()}
        return sample

class MergedDataset(Dataset):
    """
    Merges pancreas CT data from multiple source folders.
    Reads, filters, and indexes valid slices containing Class 2 (Aorta) and non-zero labels.
    """
    def __init__(self, list_path, transform=None):
        self.transform = transform
        self.index = []

        with open(list_path) as f:
            patient_dirs = [line.strip() for line in f.readlines() if line.strip()]
        
        for patient_dir in patient_dirs:
            if os.path.exists(os.path.join(patient_dir, 'label_result.npy')):
                ap_dirs = [patient_dir]
            else:
                ap_dirs = glob.glob(os.path.join(patient_dir, '**', 'AP'), recursive=True)
            for ap_dir in ap_dirs:
                label_path = os.path.join(ap_dir, 'label_result.npy')
                if not os.path.exists(label_path):
                    continue
                
                label = np.load(label_path, mmap_mode='r')
                num_slices = label.shape[2]
                
                for slice_idx in range(num_slices):
                    slice_label = label[:, :, slice_idx]
                    
                    # Skip empty labels or slices without class 2 (Aorta)
                    if not np.any(slice_label) or (2 not in slice_label):
                        continue
                    
                    self.index.append((ap_dir, slice_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ap_dir, slice_idx = self.index[idx]

        image = np.load(os.path.join(ap_dir, 'plain_result.npy'), mmap_mode='r')[:, :, slice_idx]
        label = np.load(os.path.join(ap_dir, 'label_result.npy'), mmap_mode='r')[:, :, slice_idx]

        image = np.array(image)
        label = np.array(label)

        vmin = -152.16
        vmax = 255.98

        image = np.clip(image, vmin, vmax)
        image = (image - vmin) / (vmax - vmin)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = f"{os.path.dirname(ap_dir)}_slice{slice_idx}"
        return sample
