"""Build a deterministic manifest of AP directories containing paired NPY files."""

import argparse
import glob
import os
from collections import Counter


DEFAULT_ROOTS = (
    '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/nature/Dataset/internal_2024',
    '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/nature/Dataset/internal_3',
    '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/nature/Dataset/internal_supp',
    '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/nature/Dataset/internal_xxx',
)
DEFAULT_OUTPUT = (
    '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/'
    'DataCleaning/DataLists/generate_list.txt'
)
REQUIRED_FILENAMES = ('plain_result.npy', 'label_result.npy')


def find_complete_case_directories(root_dir):
    """Find directories under one root and classify their required-file status."""
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f'Input root does not exist: {root_dir}')

    candidate_dirs = set()
    for filename in REQUIRED_FILENAMES:
        pattern = os.path.join(root_dir, '**', filename)
        candidate_dirs.update(os.path.dirname(path) for path in glob.glob(pattern, recursive=True))

    complete_dirs = []
    status_counts = Counter()
    for candidate_dir in sorted(candidate_dirs):
        plain_path = os.path.join(candidate_dir, 'plain_result.npy')
        label_path = os.path.join(candidate_dir, 'label_result.npy')
        if os.path.isfile(plain_path) and os.path.isfile(label_path):
            complete_dirs.append(os.path.abspath(candidate_dir))
            status_counts['complete'] += 1
        elif os.path.isfile(plain_path):
            status_counts['missing_label_result'] += 1
        else:
            status_counts['missing_plain_result'] += 1
    return complete_dirs, status_counts


def collect_case_directories(root_dirs):
    """Collect unique complete case directories from all requested data roots."""
    unique_dirs = {}
    root_summaries = []
    for root_dir in root_dirs:
        complete_dirs, status_counts = find_complete_case_directories(root_dir)
        duplicate_count = 0
        for case_dir in complete_dirs:
            normalized_dir = os.path.normcase(os.path.normpath(case_dir))
            if normalized_dir in unique_dirs:
                duplicate_count += 1
                continue
            unique_dirs[normalized_dir] = case_dir
        root_summaries.append((root_dir, status_counts, duplicate_count))
    return sorted(unique_dirs.values()), root_summaries


def write_manifest(output_path, case_dirs):
    """Write one complete AP directory per line in a reproducible order."""
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        for case_dir in case_dirs:
            file.write(f'{case_dir}\n')


def main():
    """Create a complete list that can be consumed directly by TestDataset."""
    parser = argparse.ArgumentParser(
        description=(
            'Recursively collect directories containing both plain_result.npy '
            'and label_result.npy.'
        )
    )
    parser.add_argument(
        '--roots',
        nargs='+',
        default=DEFAULT_ROOTS,
        help='One or more dataset roots to search recursively.',
    )
    parser.add_argument(
        '--output',
        default=DEFAULT_OUTPUT,
        help='Destination txt manifest. Each line is a complete AP directory.',
    )
    args = parser.parse_args()

    case_dirs, root_summaries = collect_case_directories(args.roots)
    if not case_dirs:
        raise RuntimeError('No directories containing both required NPY files were found.')

    write_manifest(args.output, case_dirs)
    for root_dir, status_counts, duplicate_count in root_summaries:
        print(
            f'{root_dir}: complete={status_counts["complete"]}, '
            f'missing_plain_result={status_counts["missing_plain_result"]}, '
            f'missing_label_result={status_counts["missing_label_result"]}, '
            f'duplicates_ignored={duplicate_count}'
        )
    print(f'Wrote {len(case_dirs)} complete case directories to: {args.output}')


if __name__ == '__main__':
    main()
