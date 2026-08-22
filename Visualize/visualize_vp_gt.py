"""Render VP image and ground-truth overlays from a VP manifest."""

import argparse
import os

import matplotlib
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

# Use a file-only backend so the script runs on headless training servers.
matplotlib.use('Agg')
import matplotlib.pyplot as plt


CLASS_NAMES = {
    0: 'background',
    1: 'tumour',
    2: 'aorta',
    3: 'superior mesenteric artery',
    4: 'celiac axis',
    5: 'class_5',
}
CLASS_COLORS = (
    (0.0, 0.0, 0.0, 0.0),
    (1.0, 0.15, 0.15, 0.70),
    (0.10, 0.35, 1.0, 0.70),
    (0.10, 0.80, 0.25, 0.70),
    (1.0, 0.85, 0.05, 0.70),
    (1.0, 0.10, 0.85, 0.70),
)


def read_case_dirs(list_path):
    """Read unique VP directories from a manifest file."""
    with open(list_path, 'r', encoding='utf-8') as file:
        case_dirs = [line.strip() for line in file if line.strip()]
    return sorted(set(case_dirs))


def normalize_ct(image_slice, window_center, window_width):
    """Apply a CT window and map intensities to the range [0, 1]."""
    lower_bound = window_center - window_width / 2.0
    upper_bound = window_center + window_width / 2.0
    image_slice = np.clip(image_slice.astype(np.float32), lower_bound, upper_bound)
    return (image_slice - lower_bound) / (upper_bound - lower_bound)


def choose_slice_indices(label_volume, slices_per_case):
    """Select evenly distributed slices spanning the labelled anatomy range."""
    foreground_per_slice = np.count_nonzero(label_volume, axis=(0, 1))
    foreground_indices = np.flatnonzero(foreground_per_slice)
    if foreground_indices.size == 0:
        foreground_indices = np.arange(label_volume.shape[2])
    positions = np.linspace(
        0,
        foreground_indices.size - 1,
        num=min(slices_per_case, foreground_indices.size),
        dtype=int,
    )
    return np.unique(foreground_indices[positions])


def make_colormap():
    """Create the categorical colormap used for label-only and overlay panels."""
    colormap = ListedColormap(CLASS_COLORS)
    normalization = BoundaryNorm(np.arange(-0.5, len(CLASS_COLORS) + 0.5), colormap.N)
    return colormap, normalization


def render_case(case_dir, output_dir, slices_per_case, rotation_k, window_center, window_width):
    """Render selected VP slices as raw image, label map, and blended overlay PNGs."""
    image_path = os.path.join(case_dir, 'plain_result.npy')
    label_path = os.path.join(case_dir, 'label_result.npy')
    if not (os.path.isfile(image_path) and os.path.isfile(label_path)):
        return 0

    image_volume = np.load(image_path, mmap_mode='r')
    label_volume = np.load(label_path, mmap_mode='r')
    if image_volume.shape != label_volume.shape or image_volume.ndim != 3:
        raise ValueError(
            f'Expected matching 3D image and label volumes for {case_dir}, but received '
            f'{image_volume.shape} and {label_volume.shape}.'
        )

    path_parts = os.path.normpath(case_dir).split(os.sep)
    readable_parts = path_parts[-3:] if len(path_parts) >= 3 else path_parts
    case_output_dir = os.path.join(output_dir, '__'.join(readable_parts))
    os.makedirs(case_output_dir, exist_ok=True)
    colormap, normalization = make_colormap()

    for slice_index in choose_slice_indices(label_volume, slices_per_case):
        image_slice = np.rot90(image_volume[:, :, slice_index], k=rotation_k)
        label_slice = np.rot90(label_volume[:, :, slice_index], k=rotation_k)
        display_image = normalize_ct(image_slice, window_center, window_width)
        present_labels = np.unique(label_slice).tolist()
        present_label_text = ', '.join(
            f'{label_id}:{CLASS_NAMES.get(int(label_id), "unknown")}'
            for label_id in present_labels
        )

        figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        axes[0].imshow(display_image, cmap='gray', vmin=0.0, vmax=1.0)
        axes[0].set_title('VP image')
        axes[1].imshow(label_slice, cmap=colormap, norm=normalization, interpolation='nearest')
        axes[1].set_title('Ground-truth labels')
        axes[2].imshow(display_image, cmap='gray', vmin=0.0, vmax=1.0)
        axes[2].imshow(
            np.ma.masked_equal(label_slice, 0),
            cmap=colormap,
            norm=normalization,
            interpolation='nearest',
        )
        axes[2].set_title('VP image with GT overlay')
        for axis in axes:
            axis.axis('off')
        figure.suptitle(
            f'{case_dir}\nSlice {slice_index} | labels: {present_label_text}',
            fontsize=9,
        )
        output_path = os.path.join(case_output_dir, f'slice_{slice_index:04d}.png')
        figure.savefig(output_path, dpi=160)
        plt.close(figure)

    return len(choose_slice_indices(label_volume, slices_per_case))


def main():
    """Sample VP cases from a manifest and render image/GT inspection panels."""
    parser = argparse.ArgumentParser(
        description='Visualize raw VP slices and their ground-truth labels.'
    )
    parser.add_argument('--list-dir', required=True, help='VP manifest with one directory per line.')
    parser.add_argument('--output-dir', required=True, help='Directory for generated PNG panels.')
    parser.add_argument('--num-cases', type=int, default=5, help='Number of VP cases to sample.')
    parser.add_argument('--slices-per-case', type=int, default=6)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--rotation-k', type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument('--window-center', type=float, default=40.0)
    parser.add_argument('--window-width', type=float, default=350.0)
    args = parser.parse_args()

    if args.num_cases < 1 or args.slices_per_case < 1 or args.window_width <= 0:
        raise ValueError('num-cases, slices-per-case, and window-width must be positive.')

    case_dirs = read_case_dirs(args.list_dir)
    if not case_dirs:
        raise RuntimeError(f'No VP directories were found in {args.list_dir}.')
    generator = np.random.default_rng(args.seed)
    selected_indices = generator.choice(
        len(case_dirs), size=min(args.num_cases, len(case_dirs)), replace=False
    )
    selected_case_dirs = [case_dirs[index] for index in sorted(selected_indices)]

    os.makedirs(args.output_dir, exist_ok=True)
    rendered_slice_count = 0
    for case_dir in selected_case_dirs:
        rendered_slice_count += render_case(
            case_dir,
            args.output_dir,
            args.slices_per_case,
            args.rotation_k,
            args.window_center,
            args.window_width,
        )
    print(
        f'Rendered {rendered_slice_count} VP slices from {len(selected_case_dirs)} cases '
        f'to: {args.output_dir}'
    )


if __name__ == '__main__':
    main()
