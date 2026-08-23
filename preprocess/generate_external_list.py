"""Build a phase-specific manifest from an external dataset root."""

import argparse
import glob
import os
from collections import Counter


REQUIRED_FILENAMES = ('plain_result.npy', 'label_result.npy')


def find_phase_directories(root_dir, phase):
    """Return all directories named for the requested phase beneath one root."""
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f'Input root does not exist: {root_dir}')

    pattern = os.path.join(root_dir, '**', phase)
    return sorted(
        path for path in glob.glob(pattern, recursive=True)
        if os.path.isdir(path)
    )


def collect_complete_case_directories(root_dir, phase):
    """Collect complete phase directories and count incomplete candidates."""
    complete_dirs = []
    status_counts = Counter()
    for case_dir in find_phase_directories(root_dir, phase):
        has_plain = os.path.isfile(os.path.join(case_dir, REQUIRED_FILENAMES[0]))
        has_label = os.path.isfile(os.path.join(case_dir, REQUIRED_FILENAMES[1]))
        if has_plain and has_label:
            complete_dirs.append(os.path.abspath(case_dir))
            status_counts['complete'] += 1
        elif has_plain:
            status_counts['missing_label_result'] += 1
        elif has_label:
            status_counts['missing_plain_result'] += 1
        else:
            status_counts['missing_both_required_files'] += 1

    return complete_dirs, status_counts


def collect_external_case_directories(root_dirs, phase):
    """Merge complete directories across roots while retaining per-root counts."""
    unique_dirs = {}
    root_summaries = []
    for root_dir in root_dirs:
        case_dirs, status_counts = collect_complete_case_directories(root_dir, phase)
        duplicate_count = 0
        for case_dir in case_dirs:
            normalized_dir = os.path.normcase(os.path.normpath(case_dir))
            if normalized_dir in unique_dirs:
                duplicate_count += 1
                continue
            unique_dirs[normalized_dir] = case_dir
        root_summaries.append((root_dir, status_counts, duplicate_count))
    return sorted(unique_dirs.values()), root_summaries


def write_manifest(output_path, case_dirs):
    """Write one complete phase directory per line in deterministic order."""
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        for case_dir in case_dirs:
            file.write(f'{case_dir}\n')


def main():
    """Create an external test manifest without mixing AP and VP label schemas."""
    parser = argparse.ArgumentParser(
        description=(
            'Recursively collect complete AP or VP case directories from an '
            'external dataset root.'
        )
    )
    parser.add_argument(
        '--roots',
        nargs='+',
        required=True,
        help='One or more external dataset roots to search recursively.',
    )
    parser.add_argument(
        '--phase',
        choices=('AP', 'VP'),
        default='AP',
        help='Phase directory name to include. Default: AP.',
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Destination txt manifest for the external evaluation set.',
    )
    args = parser.parse_args()

    case_dirs, root_summaries = collect_external_case_directories(
        args.roots, args.phase
    )
    if not case_dirs:
        raise RuntimeError(
            f'No complete {args.phase} directories were found under the supplied roots.'
        )

    write_manifest(args.output, case_dirs)
    for root_dir, status_counts, duplicate_count in root_summaries:
        print(
            f'{root_dir}: phase={args.phase}, complete={status_counts["complete"]}, '
            f'missing_plain_result={status_counts["missing_plain_result"]}, '
            f'missing_label_result={status_counts["missing_label_result"]}, '
            f'missing_both_required_files={status_counts["missing_both_required_files"]}, '
            f'duplicates_ignored={duplicate_count}'
        )
    print(f'Wrote {len(case_dirs)} complete {args.phase} directories to: {args.output}')


if __name__ == '__main__':
    main()
