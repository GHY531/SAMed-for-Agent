import numpy as np
import os
from tensorboardX import SummaryWriter
import re

def rotateImage(arr, k=-1):
    """
    Rotates the input image by k * 90 degrees if needed.
    """
    if hasattr(arr, 'detach'):
        arr = arr.detach().cpu().numpy()
    return np.rot90(arr, k=k).copy()

def normalize(img, window_center=40, window_width=350):
    """
    Applies CT Windowing (Window Center / Window Width) to enhance soft tissue contrast,
    then normalizes pixel intensities to [0, 255] uint8 grayscale.
    """
    if hasattr(img, 'detach'):
        img = img.detach().cpu().numpy()
        
    img = img.astype(np.float32)
    
    # Calculate window bounds
    min_hu = window_center - (window_width / 2.0)
    max_hu = window_center + (window_width / 2.0)
    
    # Clip values outside window range
    img = np.clip(img, min_hu, max_hu)
    
    # Normalize to [0, 255]
    img = (img - min_hu) / (window_width + 1e-8) * 255.0
    return img.astype(np.uint8)


def overlay_mask(ct_display, mask, color_map, alpha=0.5):
    """
    Overlays colored segmentation masks onto the grayscale CT image.
    """
    if hasattr(ct_display, 'detach'):
        ct_display = ct_display.detach().cpu().numpy()
    if hasattr(mask, 'detach'):
        mask = mask.detach().cpu().numpy()

    ct_display = np.squeeze(ct_display)
    mask = np.squeeze(mask)

    if ct_display.shape != mask.shape:
        raise ValueError(f"Shape Mismatch! ct_display: {ct_display.shape}, mask: {mask.shape}. "
                         f"Please check whether the pred_slice/gt_slice's dimension is correct!")

    base = np.stack([ct_display] * 3, axis=-1).astype(np.float32)
    overlay = base.copy()

    for cls_id, rgb in color_map.items():
        region = (mask == cls_id)
        if not np.any(region):
            continue
        
        rgb_arr = np.array(rgb, dtype=np.float32)
        overlay[region] = base[region] * (1 - alpha) + rgb_arr * alpha

    return np.clip(overlay, 0, 255).astype(np.uint8)


def build_panel(ct_slice, gt_slice, color_map, rot_k=-1, alpha=0.5, gap=10, wc=40, ww=350):
    """
    Rotates, normalizes with CT windowing, overlays mask, and stitches images into a panel.
    """
    ct_slice = rotateImage(ct_slice, k=rot_k)
    gt_slice = rotateImage(gt_slice, k=rot_k)

    # Enhance CT contrast using Window Width & Window Center
    ct_display = normalize(ct_slice, window_center=wc, window_width=ww)
    gt_overlay = overlay_mask(ct_display, gt_slice, color_map, alpha=alpha)

    h = ct_display.shape[0]
    ct_rgb = np.stack([ct_display] * 3, axis=-1)
    gap_col = np.zeros((h, gap, 3), dtype=np.uint8)

    panel = np.concatenate([ct_rgb, gap_col, gt_overlay], axis=1)
    return panel


def load_ap_dirs_from_txt(txt_path):
    """
    Reads target AP directory paths directly from a txt file for targeted visualization.
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Target txt file not found: {txt_path}")
        
    ap_dirs = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            path = line.strip()
            if path:
                ap_dirs.append(path)
                
    return sorted(list(set(ap_dirs)))


# ---------------------------- Configuration --------------------------

COLOR_MAP = {
    1: (255, 0, 0),     # Tumor
    2: (0, 0, 255),     # Aorta
    3: (0, 255, 0),     # Superior Mesenteric Artery (SMA)
    4: (255, 255, 0),   # Celiac Axis
    5: (255, 0, 255),   # Common Hepatic Artery
}

writer = SummaryWriter(logdir="/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/Visualize/Bad_Data_GT")

# Path to your txt file containing target sample directories (bad_paths or clean_paths)
target_txt_path = "/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/DataCleaning/Bad_Data/Second_Round_inspection/class_overlap_bad_data.txt"  # Replace with your actual txt file path

# Load target paths from the txt file
if os.path.exists(target_txt_path):
    ap_dirs = load_ap_dirs_from_txt(target_txt_path)
    print(f"Loaded {len(ap_dirs)} targeted samples from txt file: {target_txt_path}")
else:
    print(f"Specified txt file does not exist: {target_txt_path}")
    ap_dirs = []

# ---------------------------- Process Targeted Samples --------------------------

for idx, ap_dir in enumerate(ap_dirs, 1):
    label_path = os.path.join(ap_dir, 'label_result.npy')
    plain_path = os.path.join(ap_dir, 'plain_result.npy')

    if not (os.path.exists(label_path) and os.path.exists(plain_path)):
        print(f"[{idx}/{len(ap_dirs)}] Missing files (label_result.npy or plain_result.npy), skipping: {ap_dir}")
        continue

    patient_id = os.path.basename(os.path.dirname(ap_dir))
    dataset_source = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(ap_dir))))
    
    chinese_match = re.search(r'[\u4e00-\u9fa5]+', ap_dir)
    matched_name = chinese_match.group(0) if chinese_match else ""

    if matched_name and matched_name not in patient_id:
        tag_name = f"{dataset_source}_{matched_name}_{patient_id}/Slice"
    else:
        tag_name = f"{dataset_source}_{patient_id}/Slice"

    print(f"[{idx}/{len(ap_dirs)}] Processing slices for targeted sample: {tag_name}")

    label_vol = np.load(label_path, mmap_mode='r')
    plain_vol = np.load(plain_path, mmap_mode='r')
    
    num_slices = label_vol.shape[2]

    for slice_idx in range(num_slices):
        gt_slice = np.array(label_vol[:, :, slice_idx])
        ct_slice = np.array(plain_vol[:, :, slice_idx])
        
        # Build panel with Abdominal Window (wc=40, ww=350)
        panel = build_panel(ct_slice, gt_slice, COLOR_MAP, wc=40, ww=350)
        writer.add_image(tag_name, panel, global_step=slice_idx, dataformats='HWC')

writer.close()
print("Targeted visualization completed successfully!")