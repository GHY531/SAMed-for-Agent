import os
import numpy as np
import glob
from pathlib import Path
from PIL import Image

def MatchDataset(img_dir, label_dir):
    # 1. Read file list from directory
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.png')]

    # 2. Map dataset IDs to absolute file paths
    img_map = {
        f.replace('_0000.png', '').replace('.png', ''): os.path.join(img_dir, f) for f in img_files
    }
    label_map = {
        f.replace('_0000.png', '').replace('.png', ''): os.path.join(label_dir, f) for f in label_files
    }

    # 3. Find matching and missing files
    common_ids = sorted(list(set(img_map.keys()) & set(label_map.keys())))
    missing_labels = sorted(list(set(img_map.keys()) - set(label_map.keys())))
    missing_images = sorted(list(set(label_map.keys()) - set(img_map.keys())))

    # 4. Output summary report
    print('=' * 55)
    print('             Match Report             ')
    print('=' * 55)
    print(f'Total number of images: (imagesTr): {len(img_files)}')
    print(f'Total number of label images: {len(label_files)}')
    print(f'Successfully matched images: {len(common_ids)}')
    print(f'Has image but no label: {len(missing_labels)}')
    print(f'Has label but no image: {len(missing_images)}')

    return common_ids, img_map, label_map

def png2npy(img_dir, out_dir):
    """
    convert png images to npy images
    """
    os.makedirs(out_dir, exist_ok=True)

    for file_name in os.listdir(img_dir):
        if file_name.lower().endswith('.png'):
            img_path = os.path.join(img_dir, file_name)

            with Image.open(img_path) as img:
                img_array = np.array(img)

            base_name = os.path.splitext(file_name)[0]
            out_path = os.path.join(out_dir, f"{base_name}.npy")

            np.save(out_path, img_array)
            print(f"Successfully convert: {file_name} -> {base_name}.npy")

def AortaVisibleSamples(label_dir, txt_path):
    """Write label paths for slices containing the aorta class."""
    with open(txt_path, "w") as f:
        for label_path in Path(label_dir).glob("*.npy"):
            arr = np.load(label_path, mmap_mode="r")

            if (arr == 2).any():
                f.write(f"{label_path}\n")

    return 1

if __name__ == '__main__':
    #IMG_DIR = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/Dataset/Dataset503_PanTS_AP_2D/imagesTr'
    #LABEL_DIR = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/Dataset/Dataset503_PanTS_AP_2D/labelsTr'

    #MatchDataset(IMG_DIR, LABEL_DIR)

    #img_dir = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/Dataset/Dataset503_PanTS_AP_2D/labelsTr'
    #output_dir = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/Dataset_AP/label'
    #img_dir = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/Dataset/Dataset503_PanTS_AP_2D/imagesTr'
    #output_dir = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/Dataset_AP/image'

    #png2npy(img_dir, output_dir)

    #sample = np.load('/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/Dataset_AP/image/PanTS2D_000023_0000.npy')
    #print("Shape:", sample.shape) #[H, W]
    #print("Unique labels:", np.unique(sample)) #image 0~255

    label_dir = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/Dataset_AP/label'
    txt_path = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/Dataset_AP/positive_sample/aorta_visible_samples.txt'
    AortaVisibleSamples(label_dir, txt_path)
