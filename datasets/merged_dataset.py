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
    image = ndimage.rotate(image, angle, order=3, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size, low_res, enable_random_orientation=False):
        self.output_size = output_size
        self.low_res = low_res
        self.enable_random_orientation = enable_random_orientation

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if self.enable_random_orientation and random.random() > 0.5:
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
    Load positive cases and index their aorta-visible 2D slices.

    Tumour-present and tumour-absent refer to slices within positive cases,
    not to patient-level disease status.
    """
    def __init__(self, list_path, transform=None, rotation_k=3, num_classes=4):
        self.transform = transform
        self.index = []
        self.positive_indices = []
        self.negative_indices = []
        self.rotation_k = rotation_k
        self.num_classes = num_classes

        if rotation_k not in (0, 1, 2, 3):
            raise ValueError('rotation_k must be one of 0, 1, 2, or 3.')

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
                    
                    # Keep the anatomically defined aorta-visible slice subset.
                    if not np.any(slice_label == 2):
                        continue

                    dataset_index = len(self.index)
                    self.index.append((ap_dir, slice_idx))
                    if np.any(slice_label == 1):
                        self.positive_indices.append(dataset_index)
                    else:
                        self.negative_indices.append(dataset_index)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ap_dir, slice_idx = self.index[idx]

        image = np.load(os.path.join(ap_dir, 'plain_result.npy'), mmap_mode='r')[:, :, slice_idx]
        label = np.load(os.path.join(ap_dir, 'label_result.npy'), mmap_mode='r')[:, :, slice_idx]

        image = np.array(image)
        label = np.array(label)

        # Match the clockwise 90-degree orientation selected on validation data.
        image = np.rot90(image, k=self.rotation_k).copy()
        label = np.rot90(label, k=self.rotation_k).copy()

        # Ignore labels outside the configured model output range.
        label = np.where(
            (label >= 0) & (label <= self.num_classes),
            label,
            0,
        )

        vmin = -152.16
        vmax = 255.98

        image = np.clip(image, vmin, vmax)
        image = (image - vmin) / (vmax - vmin)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = f"{os.path.dirname(ap_dir)}_slice{slice_idx}"
        return sample
