"""Generate VP manifests from existing AP train and test manifests."""

import argparse
import glob
import os


DEFAULT_TRAIN_LIST = (
    '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/'
    'lists/lists_Pancreas/train_list.txt'
)
DEFAULT_TEST_LIST = (
    '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/'
    'lists/lists_Pancreas/test_list.txt'
)
DEFAULT_TRAIN_OUTPUT = (
    '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/'
    'lists/lists_Pancreas/train_vp_list.txt'
)
DEFAULT_TEST_OUTPUT = (
    '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/'
    'lists/lists_Pancreas/test_vp_list.txt'
)
REQUIRED_FILENAMES = ('plain_result.npy', 'label_result.npy')


def read_manifest(list_path):
    """Read non-empty source paths from an existing AP manifest."""
    with open(list_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]


def find_vp_directories(source_path):
    """Locate VP directories corresponding to one AP directory or patient path."""
    normalized_path = os.path.normpath(source_path)
    if os.path.basename(normalized_path).upper() == 'AP':
        sibling_vp = os.path.join(os.path.dirname(normalized_path), 'VP')
        return [sibling_vp] if os.path.isdir(sibling_vp) else []

    pattern = os.path.join(normalized_path, '**', 'VP')
    return sorted(glob.glob(pattern, recursive=True))


def has_required_files(vp_dir):
    """Return whether one VP directory contains both required NPY files."""
    return all(
        os.path.isfile(os.path.join(vp_dir, filename))
        for filename in REQUIRED_FILENAMES
    )


def build_vp_manifest(source_paths):
    """Collect unique, complete VP directories while silently skipping invalid ones."""
    complete_dirs = {}
    for source_path in source_paths:
        for vp_dir in find_vp_directories(source_path):
            if not has_required_files(vp_dir):
                continue
            absolute_dir = os.path.abspath(vp_dir)
            normalized_dir = os.path.normcase(os.path.normpath(absolute_dir))
            complete_dirs[normalized_dir] = absolute_dir
    return sorted(complete_dirs.values())


def write_manifest(output_path, vp_dirs):
    """Write one complete VP directory per line for direct TestDataset input."""
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        for vp_dir in vp_dirs:
            file.write(f'{vp_dir}\n')


def main():
    """Create matched VP train and test manifests from the existing AP split."""
    parser = argparse.ArgumentParser(
        description=(
            'Find complete VP directories referenced by existing AP train and '
            'test manifests. Missing VP directories or missing NPY files are skipped.'
        )
    )
    parser.add_argument('--train-list', default=DEFAULT_TRAIN_LIST)
    parser.add_argument('--test-list', default=DEFAULT_TEST_LIST)
    parser.add_argument('--train-output', default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument('--test-output', default=DEFAULT_TEST_OUTPUT)
    args = parser.parse_args()

    train_vp_dirs = build_vp_manifest(read_manifest(args.train_list))
    test_vp_dirs = build_vp_manifest(read_manifest(args.test_list))
    write_manifest(args.train_output, train_vp_dirs)
    write_manifest(args.test_output, test_vp_dirs)

    print(f'Wrote {len(train_vp_dirs)} VP train directories to: {args.train_output}')
    print(f'Wrote {len(test_vp_dirs)} VP test directories to: {args.test_output}')


if __name__ == '__main__':
    main()
