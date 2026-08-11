import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


CLASS_NAMES = {
    0: 'background',
    1: 'tumour',
    2: 'aorta',
    3: 'superior mesenteric artery',
    4: 'celiac axis',
}

def visualize_single_class(img_path, gt_path, pred_path, target_class, 
                           save_name="./results/single_class_check.png", max_slices=10):
    """
    Specify a certain organ, and it will automatically search for and render slices containing this organ
    :param target_class: class of organs ID (1, 2, 3, 4)
    :param max_slices: Maximum number of rendering layers
    """
    output_dir = os.path.dirname(save_name)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    img_data  = nib.load(img_path).get_fdata()
    gt_data   = nib.load(gt_path).get_fdata()
    pred_data = nib.load(pred_path).get_fdata()

    class_name = CLASS_NAMES.get(target_class, f"Class {target_class}")

    matching_slices = []
    for z in range(gt_data.shape[2]):
        has_gt   = np.any(gt_data[:, :, z] == target_class)
        has_pred = np.any(pred_data[:, :, z] == target_class)
        if has_gt or has_pred:
            matching_slices.append(z)

    num_found = len(matching_slices)
    if num_found == 0:
        print(f"Attention please : In this NIfTI file, there's no found of GT and Pred of Class {target_class}: {class_name}!")
        return

    if num_found > max_slices:
        step = num_found // max_slices
        selected_slices = matching_slices[::step][:max_slices]
    else:
        selected_slices = matching_slices

    num_rows = len(selected_slices)

    overlay_cmap = mcolors.ListedColormap([(1.0, 0.2, 0.2, 0.7)])

    fig, axes = plt.subplots(num_rows, 3, figsize=(12, 3.8 * num_rows))
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, z in enumerate(selected_slices):
        img_s  = img_data[:, :, z]
        gt_mask   = (gt_data[:, :, z] == target_class)
        pred_mask = (pred_data[:, :, z] == target_class)

        axes[i, 0].imshow(img_s, cmap='gray')
        axes[i, 0].set_ylabel(f"Slice {z}", fontsize=11, fontweight='bold')
        if i == 0:
            axes[i, 0].set_title("CT Image", fontsize=13, fontweight='bold')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        axes[i, 1].imshow(img_s, cmap='gray')
        axes[i, 1].imshow(np.ma.masked_where(~gt_mask, gt_mask), cmap=overlay_cmap)
        if i == 0:
            axes[i, 1].set_title(f"GT: {class_name}", fontsize=13, fontweight='bold', color='darkred')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(img_s, cmap='gray')
        axes[i, 2].imshow(np.ma.masked_where(~pred_mask, pred_mask), cmap=overlay_cmap)
        if i == 0:
            axes[i, 2].set_title(f"Pred: {class_name}", fontsize=13, fontweight='bold', color='darkred')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Successfully found {num_found} slices including {class_name}")
    print(f"Successfully selected {len(selected_slices)} layers and save them to: {save_name}")

visualize_single_class(
    '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/TestResult/predictions/张旦_img.nii.gz',
    '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/TestResult/predictions/张旦_gt.nii.gz', 
    '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/TestResult/predictions/张旦_pred.nii.gz', 
    target_class=1,
    save_name='/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/TestResult/VisualizedImage'
)