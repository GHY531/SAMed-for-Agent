import os
import torch
import numpy as np
from torch.utils.data import Dataset
import glob


class TestDataset(Dataset):
    """
    To test trained SAMed
    """
    def __init__(self, list_path, transform=None):
        self.transform = transform
        self.cases = []

        with open(list_path) as f:
            patient_dirs = [line.strip() for line in f.readlines() if line.strip()]

        for patient_dir in patient_dirs:

            # 如果txt直接给AP目录
            if os.path.exists(os.path.join(patient_dir, 'label_result.npy')):
                ap_dirs = [patient_dir]

            # 如果txt给病人目录，递归找AP
            else:
                ap_dirs = glob.glob(
                    os.path.join(patient_dir, '**', 'AP'),
                    recursive=True
                )

            for ap_dir in ap_dirs:

                label_path = os.path.join(
                    ap_dir,
                    'label_result.npy'
                )

                if not os.path.exists(label_path):
                    continue

                label = np.load(
                    label_path,
                    mmap_mode='r'
                )

                if np.any(label) and np.any(label == 2):
                    self.cases.append(label_path)


    def __len__(self):
        return len(self.cases)


    def __getitem__(self, idx):

        label_path = self.cases[idx]

        image_path = os.path.join(
            os.path.dirname(label_path),
            'plain_result.npy'
        )

        image = np.load(
            image_path,
            mmap_mode="r"
        )

        label = np.load(
            label_path,
            mmap_mode="r"
        )

        image = np.array(image, dtype=np.float32)
        label = np.array(label)

        vmin = -152.16
        vmax = 255.98

        image = np.clip(image, vmin, vmax)
        image = (image - vmin) / (vmax - vmin)

        image = np.moveaxis(image, 2, 0)
        label = np.moveaxis(label, 2, 0)

        sample = {
            'image': torch.from_numpy(image).float(),
            'label': torch.from_numpy(label).float(),
        }

        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = os.path.splitext(
            os.path.basename(label_path)
        )[0]

        return sample