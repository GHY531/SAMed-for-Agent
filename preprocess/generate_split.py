import json
import os
import random

import numpy as np

INPUT_PATH = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/Dataset_AP/positive_sample/aorta_visible_samples.txt'
OUTPUT_DIR = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/Dataset_AP/positive_sample'
TRAIN_FRACTION = 0.8


def read_unique_paths(input_path):
    """Read non-empty label paths and reject duplicate entries."""
    with open(input_path, 'r', encoding='utf-8') as file:
        paths = [line.strip() for line in file if line.strip()]

    if len(paths) != len(set(paths)):
        raise ValueError(f'Duplicate label paths found in {input_path}.')
    if not paths:
        raise ValueError(f'No label paths found in {input_path}.')
    return paths

def stratify_by_tumour(label_paths):
    """Separate aorta-visible slices by the presence of tumour class 1."""
    positive_paths = []
    negative_paths = []
    for label_path in label_paths:
        if not os.path.isfile(label_path):
            raise FileNotFoundError(f'Label file does not exist: {label_path}')
        label = np.load(label_path, mmap_mode='r')
        if np.any(label == 1):
            positive_paths.append(label_path)
        else:
            negative_paths.append(label_path)
    return positive_paths, negative_paths


def split_stratum(paths, train_fraction):
    """Shuffle one stratum and split it with a fixed training fraction."""
    shuffled_paths = paths.copy()
    random.shuffle(shuffled_paths)
    split_index = int(len(shuffled_paths) * train_fraction)
    return shuffled_paths[:split_index], shuffled_paths[split_index:]


def write_paths(output_path, paths):
    """Write one path per line using a stable text encoding."""
    with open(output_path, 'w', encoding='utf-8') as file:
        for path in paths:
            file.write(f'{path}\n')


def build_summary(train_paths, test_paths, positive_set):
    """Build auditable statistics for the generated manifests."""
    train_positive = sum(path in positive_set for path in train_paths)
    test_positive = sum(path in positive_set for path in test_paths)
    return {
        'seed': 16,
        'train_fraction': TRAIN_FRACTION,
        'split_level': 'slice',
        'patient_grouping_available': False,
        'train': {
            'total': len(train_paths),
            'tumour_positive': train_positive,
            'tumour_negative': len(train_paths) - train_positive,
        },
        'test': {
            'total': len(test_paths),
            'tumour_positive': test_positive,
            'tumour_negative': len(test_paths) - test_positive,
        },
    }


def main():
    """Generate reproducible tumour-stratified train and test manifests."""
    all_slices = read_unique_paths(INPUT_PATH)
    positive_slices, negative_slices = stratify_by_tumour(all_slices)

    print(f'Total slices read: {len(all_slices)}')
    print(f'Tumour-positive slices: {len(positive_slices)}')
    print(f'Tumour-negative slices: {len(negative_slices)}')
    if not positive_slices or not negative_slices:
        raise ValueError('Both tumour-positive and tumour-negative slices are required.')

    random.seed(16) #Salute to my favorite Formula 1 driver, Charles LECLERC
                    #For his wonderful driving in SilverStone and Spa
                    #Not today, Not tomorrow, But one day, He will be WDC!!!
    train_positive, test_positive = split_stratum(
        positive_slices,
        TRAIN_FRACTION,
    )
    train_negative, test_negative = split_stratum(
        negative_slices,
        TRAIN_FRACTION,
    )

    train_slices = train_positive + train_negative
    test_slices = test_positive + test_negative
    random.shuffle(train_slices)
    random.shuffle(test_slices)

    overlap = set(train_slices) & set(test_slices)
    if overlap:
        raise RuntimeError(f'Train/test overlap detected for {len(overlap)} slices.')
    if len(train_slices) + len(test_slices) != len(all_slices):
        raise RuntimeError('The generated split does not preserve every input slice.')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_paths(os.path.join(OUTPUT_DIR, 'train.txt'), train_slices)
    write_paths(os.path.join(OUTPUT_DIR, 'test.txt'), test_slices)

    summary = build_summary(train_slices, test_slices, set(positive_slices))
    summary_path = os.path.join(OUTPUT_DIR, 'split_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)
        file.write('\n')

    print(json.dumps(summary, indent=2))
    print(f'Split summary saved to: {summary_path}')


if __name__ == '__main__':
    main()
