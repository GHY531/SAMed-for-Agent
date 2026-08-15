import argparse
import os

import matplotlib

matplotlib.use('Agg')

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


CLASS_NAMES = {
    1: 'Tumour',
    2: 'Aorta',
    3: 'Superior mesenteric artery',
    4: 'Celiac axis',
}
CLASS_COLORS = {
    1: '#FF2020',
    2: '#00E5FF',
    3: '#FFD600',
    4: '#E040FB',
}


def parse_args():
    """Parse a low-Dice triplet list and batch rendering options."""
    parser = argparse.ArgumentParser(
        description='Render low-Dice 3D NIfTI triplets into one PNG per case.'
    )
    parser.add_argument(
        'sample_list',
        help='Tab-separated image, ground-truth, and prediction paths from test_3d.py',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Directory for rendered PNG files (default: next to the sample list)',
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.72,
        help='Opacity of non-background label overlays',
    )
    parser.add_argument('--dpi', type=int, default=120)
    return parser.parse_args()


def load_volume(path):
    """Load a NIfTI volume in the [Z, H, W] array order used by SimpleITK."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Required NIfTI file does not exist: {path}')
    volume = sitk.GetArrayFromImage(sitk.ReadImage(path))
    if volume.ndim != 3:
        raise ValueError(f'Expected a 3D volume, got shape {volume.shape}: {path}')
    return volume


def load_sample_paths(sample_list):
    """Load tab-separated image, ground-truth, and prediction path triplets."""
    if not os.path.isfile(sample_list):
        raise FileNotFoundError(f'Sample list does not exist: {sample_list}')

    samples = []
    with open(sample_list, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            paths = stripped_line.split('\t')
            if len(paths) != 3:
                raise ValueError(
                    f'Line {line_number} must contain exactly three tab-separated paths.'
                )
            samples.append(tuple(paths))
    return samples


def load_sample(image_path, ground_truth_path, prediction_path):
    """Load and validate one image, ground-truth, and prediction triplet."""
    paths = {
        'image': image_path,
        'ground_truth': ground_truth_path,
        'prediction': prediction_path,
    }
    volumes = {name: load_volume(path) for name, path in paths.items()}
    reference_shape = volumes['image'].shape
    mismatched_shapes = {
        name: volume.shape
        for name, volume in volumes.items()
        if volume.shape != reference_shape
    }
    if mismatched_shapes:
        raise ValueError(
            f'All case volumes must have shape {reference_shape}, '
            f'but found: {mismatched_shapes}'
        )
    return volumes


def get_tumour_slices(ground_truth, prediction):
    """Return slices containing tumour in either ground truth or prediction."""
    tumour_slices = np.any(
        (ground_truth == 1) | (prediction == 1), axis=(1, 2)
    )
    selected_indices = np.flatnonzero(tumour_slices).astype(int).tolist()
    if not selected_indices:
        raise ValueError('Neither ground truth nor prediction contains tumour slices.')
    return selected_indices


def normalize_image(image):
    """Normalize one image slice with robust percentiles for display."""
    finite_mask = np.isfinite(image)
    if not np.any(finite_mask):
        return np.zeros_like(image, dtype=np.float32)
    low, high = np.percentile(image[finite_mask], (1, 99))
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    normalized = (image.astype(np.float32) - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0)


def rotate_counterclockwise(array):
    """Rotate a 2D array counterclockwise by 90 degrees."""
    return np.rot90(array, k=1)


def build_label_colormap(alpha):
    """Create a high-contrast label map with a fully transparent background."""
    colors = [(0.0, 0.0, 0.0, 0.0)]
    for class_index in range(1, max(CLASS_NAMES) + 1):
        red, green, blue = mcolors.to_rgb(CLASS_COLORS[class_index])
        colors.append((red, green, blue, alpha))
    colormap = mcolors.ListedColormap(colors)
    boundaries = np.arange(-0.5, len(colors) + 0.5, 1.0)
    normalization = mcolors.BoundaryNorm(boundaries, colormap.N)
    return colormap, normalization


def draw_overlay(axis, image, labels, title, colormap, normalization):
    """Draw a grayscale image with transparent-background class labels."""
    axis.imshow(image, cmap='gray', vmin=0.0, vmax=1.0)
    masked_labels = np.ma.masked_where(labels == 0, labels)
    axis.imshow(masked_labels, cmap=colormap, norm=normalization)
    axis.set_title(title, fontsize=9, fontweight='bold')
    axis.axis('off')


def class_dice(ground_truth, prediction, class_index):
    """Calculate Dice for one class on the displayed slice."""
    gt_mask = ground_truth == class_index
    pred_mask = prediction == class_index
    denominator = int(gt_mask.sum() + pred_mask.sum())
    if denominator == 0:
        return 1.0
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    return float(2.0 * intersection / denominator)


def visualize_sample(
    image_path,
    ground_truth_path,
    prediction_path,
    output_path,
    alpha=0.72,
    dpi=120,
):
    """Render tumour-relevant slices from one saved 3D triplet into a PNG."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f'--alpha must be within [0, 1], got {alpha}')
    volumes = load_sample(image_path, ground_truth_path, prediction_path)
    selected_indices = get_tumour_slices(
        volumes['ground_truth'], volumes['prediction']
    )

    colormap, normalization = build_label_colormap(alpha)
    row_count = len(selected_indices)
    columns_per_slice = 3
    figure, axes = plt.subplots(
        row_count,
        columns_per_slice,
        figsize=(12, max(3.6, row_count * 3.6)),
        squeeze=False,
    )
    supported_labels = [0, *CLASS_NAMES.keys()]

    for display_index, selected_index in enumerate(selected_indices):
        row = display_index
        column_start = 0
        image = rotate_counterclockwise(
            normalize_image(volumes['image'][selected_index])
        )
        ground_truth = rotate_counterclockwise(
            volumes['ground_truth'][selected_index].astype(np.int16)
        )
        prediction = rotate_counterclockwise(
            volumes['prediction'][selected_index].astype(np.int16)
        )
        # Hide labels outside classes 1-4 without modifying the source volumes.
        ground_truth = np.where(
            np.isin(ground_truth, supported_labels),
            ground_truth,
            0,
        )
        prediction = np.where(
            np.isin(prediction, supported_labels),
            prediction,
            0,
        )

        image_axis = axes[row, column_start]
        image_axis.imshow(image, cmap='gray', vmin=0.0, vmax=1.0)
        image_axis.set_title(f'Z={selected_index} | Image', fontsize=9)
        image_axis.axis('off')
        draw_overlay(
            axes[row, column_start + 1],
            image,
            ground_truth,
            'Ground truth',
            colormap,
            normalization,
        )
        tumour_is_present = np.any(ground_truth == 1) or np.any(prediction == 1)
        if tumour_is_present:
            prediction_title = (
                f'Prediction | tumour Dice='
                f'{class_dice(ground_truth, prediction, 1):.3f}'
            )
        else:
            prediction_title = 'Prediction | tumour absent'
        draw_overlay(
            axes[row, column_start + 2],
            image,
            prediction,
            prediction_title,
            colormap,
            normalization,
        )

    legend_handles = [
        mpatches.Patch(color=CLASS_COLORS[index], label=CLASS_NAMES[index])
        for index in sorted(CLASS_NAMES)
    ]
    figure.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=len(legend_handles),
        frameon=False,
        fontsize=10,
    )
    case_name = os.path.basename(os.path.dirname(os.path.abspath(image_path)))
    figure.suptitle(
        f'{case_name} | {len(selected_indices)} tumour-relevant slices | '
        'rotated 90 degrees counterclockwise',
        fontsize=14,
        fontweight='bold',
    )
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.96), pad=0.5)

    output_path = os.path.abspath(output_path)
    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(figure)
    print(f'Visualized original Z slices: {selected_indices}')
    print(f'Saved visualization: {output_path}')
    return output_path


def visualize_samples(sample_list, output_dir, alpha=0.72, dpi=120):
    """Render every low-Dice triplet in a list into a separate PNG file."""
    samples = load_sample_paths(sample_list)
    if not samples:
        raise ValueError(f'No low-Dice samples were listed in: {sample_list}')

    os.makedirs(output_dir, exist_ok=True)
    for image_path, ground_truth_path, prediction_path in samples:
        case_name = os.path.basename(os.path.dirname(os.path.abspath(image_path)))
        output_path = os.path.join(output_dir, f'{case_name}.png')
        visualize_sample(
            image_path,
            ground_truth_path,
            prediction_path,
            output_path,
            alpha=alpha,
            dpi=dpi,
        )


def main():
    """Run the batched low-Dice 3D visualization command."""
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.sample_list)),
            'low_dice_visualizations',
        )
    visualize_samples(
        args.sample_list,
        output_dir,
        alpha=args.alpha,
        dpi=args.dpi,
    )


if __name__ == '__main__':
    main()
