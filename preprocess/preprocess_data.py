import numpy as np
import os
from scipy.ndimage import zoom
from PIL import Image

def get_max_dimensions(img_txt):
    """
    Find maximum height and width across all images in the txt file.
    """
    max_h = 0
    max_w = 0
    exceed_512_count = 0
    total_count = 0

    with open(img_txt, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in lines:
        parts = line.replace(',', ' ').split()
        img_path = parts[0]

        if not os.path.exists(img_path):
            print(f"Warning: File not found -> {img_path}")
            continue

        total_count += 1

        # Retrieve image shape efficiently without full data loading
        if img_path.endswith('.npy'):
            # Load header only via memory mapping
            shape = np.load(img_path, mmap_mode='r').shape
            h, w = shape[0], shape[1]
        else:
            # PIL open reads image metadata without loading full pixel buffer
            with Image.open(img_path) as img:
                w, h = img.size  # PIL returns (width, height)

        if h > max_h:
            max_h = h
        if w > max_w:
            max_w = w

        if h > 512 or w > 512:
            exceed_512_count += 1

    print("===== Dataset Dimension Statistics =====")
    print(f"Total images checked: {total_count}")
    print(f"Max Height (H): {max_h}")
    print(f"Max Width  (W): {max_w}")
    print(f"Images exceeding 512x512: {exceed_512_count}")

    return max_h, max_w

def process_2d_array(array, is_label=False, target_size=(512, 512)):
    """
    Proportionally scale 2D numpy array based on the long edge to preserve aspect ratio,
    then center-pad with zeros to target_size.
    """
    h, w = array.shape
    target_h, target_w = target_size

    scale = min(target_h / h, target_w / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    # Order 0 for label (preserve categorical IDs), Order 3 for image
    order = 0 if is_label else 3
    resized = zoom(array, (new_h / h, new_w / w), order=order)

    # Actual dimensions after rounding
    rh, rw = resized.shape

    # Calculate zero padding bounds
    pad_top = (target_h - rh) // 2
    pad_bottom = target_h - rh - pad_top
    pad_left = (target_w - rw) // 2
    pad_right = target_w - rw - pad_left

    return np.pad(
        resized, 
        ((pad_top, pad_bottom), (pad_left, pad_right)), 
        mode='constant', 
        constant_values=0
    )


def pad_to_square(img_txt, target_dir, target_size=(512, 512)):
    """
    Batch process 2D images and labels listed in img_txt to square .npy files.
    Constructs the image path by appending '_0000' before the extension.
    """
    out_img_dir = os.path.join(target_dir, 'image')
    out_lbl_dir = os.path.join(target_dir, 'label')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    with open(img_txt, 'r') as f:
        label_paths = [line.strip() for line in f.readlines() if line.strip()]

    for lbl_path in label_paths:
        if not os.path.exists(lbl_path):
            print(f"Warning: Label file not found -> {lbl_path}")
            continue

        # Split path into directory and filename
        lbl_dir, lbl_filename = os.path.split(lbl_path)
        base_name, ext = os.path.splitext(lbl_filename)

        # Build corresponding image directory and image filename with '_0000'
        img_dir = lbl_dir.replace('/label', '/image')
        img_filename = f"{base_name}_0000{ext}"
        img_path = os.path.join(img_dir, img_filename)

        # 1. Process Label
        lbl = np.load(lbl_path) if lbl_path.endswith('.npy') else np.array(Image.open(lbl_path))
        padded_lbl = process_2d_array(lbl, is_label=True, target_size=target_size)
        # Save label using clean base_name (e.g. PanTS2D_017438.npy)
        np.save(os.path.join(out_lbl_dir, f"{base_name}.npy"), padded_lbl)

        # 2. Process Image
        if os.path.exists(img_path):
            img = np.load(img_path) if img_path.endswith('.npy') else np.array(Image.open(img_path))
            padded_img = process_2d_array(img, is_label=False, target_size=target_size)
            # Save image using the same base_name so image and label IDs match perfectly
            np.save(os.path.join(out_img_dir, f"{base_name}.npy"), padded_img)
            print(f"Successfully processed pair: {base_name}.npy")
        else:
            print(f"Warning: Image file missing at expected path -> {img_path}")

#img_txt1 = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/Dataset_AP/positive_sample/train.txt'
#target_dir1 = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/Dataset_AP/positive_sample/train'

#img_txt2 = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/Dataset_AP/positive_sample/test.txt'
#target_dir2 = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/Dataset_AP/positive_sample/test'

#pad_to_square(img_txt1, target_dir1)
#pad_to_square(img_txt2, target_dir2)

#test_path = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/Dataset_AP/positive_sample/test/image/PanTS2D_000435.npy'
#test = np.load(test_path)
#print(test.shape)