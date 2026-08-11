import os
import torch
import numpy as np
from torch.utils.data import Dataset

class TestDataset(Dataset):
    """
    Dataset loader for 2D/3D test data stored in separate image and label directories.
    """
    def __init__(self, test_dir, transform=None):
        self.transform = transform
        self.image_dir = os.path.join(test_dir, 'image')
        self.label_dir = os.path.join(test_dir, 'label')
        self.cases = []

        if not os.path.exists(self.image_dir) or not os.path.exists(self.label_dir):
            raise ValueError(f"Directories 'image' and 'label' must exist in {test_dir}")

        # Scan image directory and pair with corresponding label file
        image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.npy')])
        for file_name in image_files:
            label_path = os.path.join(self.label_dir, file_name)
            if os.path.exists(label_path):
                self.cases.append(file_name)
            else:
                print(f"Warning: Missing matching label file for {file_name}")

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        file_name = self.cases[idx]
        image_path = os.path.join(self.image_dir, file_name)
        label_path = os.path.join(self.label_dir, file_name)

        image = np.load(image_path)
        label = np.load(label_path)

        image = np.array(image, dtype=np.float32)
        label = np.array(label, dtype=np.float32)

        # Normalize pixel values to [0, 1] if required
        if image.max() > 1.0:
            image = image / 255.0

        # Adjust dimensions to Channel-First format (C, H, W) if array is 3D (H, W, C)
        if image.ndim == 3:
            image = np.moveaxis(image, -1, 0)
        if label.ndim == 3:
            label = np.moveaxis(label, -1, 0)

        sample = {
            'image': torch.from_numpy(image).float(),
            'label': torch.from_numpy(label).float(),
        }

        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = os.path.splitext(file_name)[0]

        return sample