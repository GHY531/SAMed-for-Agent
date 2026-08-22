import os
import numpy as np
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset, Sampler
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
        self.positive_indices = []
        self.negative_indices = []

        with open(label_path) as f:
            label_paths = [line.strip() for line in f.readlines() if line.strip()]
        
        for label_path in label_paths:
            if not os.path.exists(label_path):
                continue

            label = np.load(label_path, mmap_mode='r')

            if np.any(label == 2):
                dataset_index = len(self.index)
                self.index.append(label_path)
                if np.any(label == 1):
                    self.positive_indices.append(dataset_index)
                else:
                    self.negative_indices.append(dataset_index)

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


class TumourBalancedBatchSampler(Sampler):
    """Yield batches whose epoch-level positive ratio follows a target ratio."""

    def __init__(
        self,
        positive_indices,
        negative_indices,
        batch_size,
        negative_per_positive=2,
        seed=1234,
    ):
        if negative_per_positive < 1:
            raise ValueError('negative_per_positive must be at least 1.')
        samples_per_group = negative_per_positive + 1
        if batch_size < samples_per_group:
            raise ValueError(
                'batch_size must be at least negative_per_positive + 1.'
            )
        if not positive_indices:
            raise ValueError('At least one tumour-positive index is required.')
        if not negative_indices:
            raise ValueError('At least one tumour-negative index is required.')

        self.positive_indices = list(positive_indices)
        self.negative_indices = list(negative_indices)
        self.batch_size = batch_size
        self.negative_per_positive = negative_per_positive
        self.seed = seed
        self.epoch = 0
        self.num_batches = (
            len(self.positive_indices) * samples_per_group + batch_size - 1
        ) // batch_size
        self.positive_counts_per_batch = [
            ((batch_index + 1) * batch_size) // samples_per_group
            - (batch_index * batch_size) // samples_per_group
            for batch_index in range(self.num_batches)
        ]
        self.positive_count_per_epoch = sum(self.positive_counts_per_batch)
        self.negative_count_per_epoch = (
            self.num_batches * batch_size - self.positive_count_per_epoch
        )

    def set_epoch(self, epoch):
        """Select the deterministic shuffle sequence for one epoch."""
        self.epoch = epoch

    @staticmethod
    def _sample_indices(indices, sample_count, generator):
        """Sample shuffled indices, cycling only when the pool is exhausted."""
        sampled_indices = []
        while len(sampled_indices) < sample_count:
            shuffled_indices = indices.copy()
            generator.shuffle(shuffled_indices)
            remaining = sample_count - len(sampled_indices)
            sampled_indices.extend(shuffled_indices[:remaining])
        return sampled_indices

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        positive_counts_per_batch = self.positive_counts_per_batch.copy()
        generator.shuffle(positive_counts_per_batch)
        sampled_positive = self._sample_indices(
            self.positive_indices,
            self.positive_count_per_epoch,
            generator,
        )
        sampled_negative = self._sample_indices(
            self.negative_indices,
            self.negative_count_per_epoch,
            generator,
        )

        positive_start = 0
        negative_start = 0
        for batch_index in range(self.num_batches):
            positive_count = positive_counts_per_batch[batch_index]
            negative_count = self.batch_size - positive_count
            batch = sampled_positive[
                positive_start:positive_start + positive_count
            ]
            batch += sampled_negative[
                negative_start:negative_start + negative_count
            ]
            positive_start += positive_count
            negative_start += negative_count
            generator.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches
