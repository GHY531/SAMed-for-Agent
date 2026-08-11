import os
import glob
import numpy as np

def LabelExistCheck(
                    txt_path, 
                    Bad_Data_txt_path, 
                    expected_classes=[0, 1, 2, 3, 4, 5]
                    ):
    """
    First Round Check: All train or test cases should have all labels!
    """
    expected_set = set(expected_classes)
    bad_records = {}

    with open(txt_path) as f:
        patient_dirs = [line.strip() for line in f.readlines()]
    
    for patient_dir in patient_dirs:
        ap_dirs = glob.glob(os.path.join(patient_dir, '**', 'AP'), recursive=True)
        for ap_dir in ap_dirs:
            label_path = os.path.join(ap_dir, 'label_result.npy')
            
            if not os.path.exists(label_path):
                continue
            label = np.load(label_path, mmap_mode = 'r')
            present_classes = set(np.unique(label))
            
            missing = sorted(list(expected_set - present_classes))
            
            if missing:
                if len(missing) == len(expected_set):
                    bad_records[patient_dir] = "All empty!"
                else:
                    bad_records[patient_dir] = f"missing labels: {missing}"

    with open(Bad_Data_txt_path, 'w', encoding='utf-8') as f:
        for bad_dir, detail in bad_records.items():
            f.write(f"{bad_dir}\t{detail}\n")

    print(f"Scanning finished! Found {len(bad_records)} bad cases")
    return 1

def read_patient_dirs(clean_txt_path):
    with open(clean_txt_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def iter_ap_dirs(clean_txt_path):
    for patient_dir in read_patient_dirs(clean_txt_path):
        for ap_dir in glob.glob(os.path.join(patient_dir, '**', 'AP'), recursive=True):
            label_path = os.path.join(ap_dir, 'label_result.npy')
            if os.path.exists(label_path):
                yield ap_dir, label_path

def get_z_axis_and_num_slices(shape):
    z_axis = 2
    num_slices = shape[z_axis]
    return z_axis, num_slices

def write_bad_paths(bad_txt_path, bad_paths):
    with open(bad_txt_path, 'w', encoding='utf-8') as f:
        for path in bad_paths:
            f.write(f"{path}\n")

def CheckTruncation(clean_txt_path, bad_txt_path):
    """
    step1: Head & Tail truncation check
    """
    bad_paths = []
    for ap_dir, label_path in iter_ap_dirs(clean_txt_path):
        try:
            label = np.load(label_path, mmap_mode='r')
            z_axis, num_slices = get_z_axis_and_num_slices(label.shape)

            first_slice = np.take(label, 0, axis=z_axis)
            last_slice = np.take(label, num_slices - 1, axis=z_axis)
            if np.all(first_slice == 0) or np.all(last_slice == 0):
                bad_paths.append(ap_dir)
        except Exception:
            bad_paths.append(ap_dir)

    write_bad_paths(bad_txt_path, bad_paths)
    print(f"Head/Tail truncation check done. Found {len(bad_paths)} abnormal dirs.")
    return bad_paths

def check_middle_holes(clean_txt_path, bad_txt_path, min_peak_voxels=100, max_hole_width=1):
    """
    step2: Middle holes check
    """
    bad_paths = []
    for ap_dir, label_path in iter_ap_dirs(clean_txt_path):
        try:
            label = np.load(label_path, mmap_mode='r')
            z_axis, num_slices = get_z_axis_and_num_slices(label.shape)
            axes = tuple(i for i in range(label.ndim) if i !=z_axis)
            present_classes = [c for c in np.unique(label) if c != 0]

            found = False
            for cls in present_classes:
                slice_areas = np.sum(label == cls, axis=axes)
                non_zero_zs = np.where(slice_areas > 0)[0]
                if len(non_zero_zs) == 0:
                    continue
                if np.max(slice_areas) < min_peak_voxels:
                    continue
                z_start, z_end = non_zero_zs[0], non_zero_zs[-1]

                middle_areas = slice_areas[z_start:z_end + 1]
                zero_indices = np.where(middle_areas == 0)[0]
                if len(zero_indices) > 0:
                    splits = np.where(np.diff(zero_indices) > 1)[0] + 1
                    hole_groups = np.split(zero_indices, splits)
                    if any(len(g) > max_hole_width for g in hole_groups):
                        found = True
                        break
            
            if found:
                bad_paths.append(ap_dir)
        except Exception:
            bad_paths.append(ap_dir)

    write_bad_paths(bad_txt_path, bad_paths)
    print(f"Middle hole check done. Found {len(bad_paths)} abnormal dirs.")
    return bad_paths
    
import numpy as np

def check_class_overlap(clean_txt_path, bad_txt_path, overlap_voxel_threshold=30):
    """
    step3: Overlap check
    """
    bad_paths = []
    for ap_dir, label_path in iter_ap_dirs(clean_txt_path):
        try:
            label = np.load(label_path, mmap_mode='r')
            # Convert memmap to numpy array to prevent np.take slicing errors
            label = np.asarray(label)
            
            z_axis, num_slices = get_z_axis_and_num_slices(label.shape)
            present_classes = [c for c in np.unique(label) if c != 0]

            found = False
            for i in range(len(present_classes)):
                for j in range(i + 1, len(present_classes)):
                    cls1, cls2 = present_classes[i], present_classes[j]
                    for z in range(num_slices - 1):
                        mask_z = np.take(label, z, axis=z_axis)
                        mask_z1 = np.take(label, z + 1, axis=z_axis)

                        overlap_fwd = np.sum((mask_z == cls1) & (mask_z1 == cls2))
                        overlap_bwd = np.sum((mask_z == cls2) & (mask_z1 == cls1))
                        if overlap_fwd > overlap_voxel_threshold or overlap_bwd > overlap_voxel_threshold:
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

            if found:
                bad_paths.append(ap_dir)
        except Exception as e:
            # Print error to avoid losing track of missing/corrupted files during manual review
            print(f"Error loading {ap_dir}: {e}")
            bad_paths.append(ap_dir)
    
    write_bad_paths(bad_txt_path, bad_paths)
    print(f"Class overlap check done. Found {len(bad_paths)} abnormal dirs.")
    return bad_paths

def check_area_jump(clean_txt_path, bad_txt_path, min_peak_voxels=100, area_jump_ratio=0.6, min_jump_voxels=100):
    """
    step4: Area Jump check
    """
    bad_paths = []

    def is_jump(a1, a2):
        max_a = max(a1, a2)
        if max_a == 0:
            return False
        diff = abs(a1-a2)
        return (diff / max_a) > area_jump_ratio and diff > min_jump_voxels

    for ap_dir, label_path in iter_ap_dirs(clean_txt_path):
        try:
            label = np.load(label_path, mmap_mode='r')
            z_axis, num_slices = get_z_axis_and_num_slices(label.shape)
            axes = tuple(i for i in range(label.ndim) if i!=z_axis)
            present_classes = [c for c in np.unique(label) if c!=0]

            found = False
            for cls in present_classes:
                slice_areas = np.sum(label == cls, axis=axes)
                non_zero_zs = np.where(slice_areas > 0)[0]
                if len(non_zero_zs) == 0:
                    continue
                if np.max(slice_areas) < min_peak_voxels:
                    continue
                z_start, z_end = non_zero_zs[0], non_zero_zs[-1]

                if z_start > 0 and is_jump(0, slice_areas[z_start]):
                    found = True
                    break
                if z_end < num_slices - 1 and is_jump(slice_areas[z_end], 0):
                    found = True
                    break
                for z in range(z_start, z_end):
                    a1, a2 = slice_areas[z], slice_areas[z+1]
                    if a1 > 0 and a2 > 0 and is_jump(a1, a2):
                        found = True
                        break
                if found:
                    break
            
            if found:
                bad_paths.append(ap_dir)
        except Exception:
            bad_paths.append(ap_dir)
    write_bad_paths(bad_txt_path, bad_paths)
    print(f"Area Jump check done. Found {len(bad_paths)} abnormal dirs.")

def load_patients_paths(txt_path):
    bad_paths = set()
    if not os.path.exists(txt_path):
        return bad_paths

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                raw_path = line.split('\t')[0]
                norm_path = os.path.normpath(raw_path)
                bad_paths.add(norm_path)

    return bad_paths

def generate_clean_patient_list(txt_path, bad_txt_path, clean_txt_path):
    # Load all bad AP paths into a normalized set
    bad_paths = load_patients_paths(bad_txt_path)

    with open(txt_path, 'r', encoding='utf-8') as f:
        base_patient_paths = [line.strip() for line in f.readlines() if line.strip()]

    clean_patients_paths = []
    remove_count = 0
    total_ap_count = 0

    for base_path in base_patient_paths:
        if not os.path.exists(base_path):
            print(f"Path does not exist, skipping: {base_path}")
            continue

        # Search recursively for all AP subdirectories under this base path
        found_ap_dirs = glob.glob(os.path.join(base_path, "**", "AP"), recursive=True)
        total_ap_count += len(found_ap_dirs)

        for ap_path in found_ap_dirs:
            norm_path = os.path.normpath(ap_path)
            # Filter out paths that exist in bad_paths
            if norm_path in bad_paths:
                remove_count += 1
            else:
                clean_patients_paths.append(ap_path)

    # Overwrite clean_txt_path with mode 'w' to prevent duplicate appending
    with open(clean_txt_path, 'w', encoding='utf-8') as f:
        for path in clean_patients_paths:
            f.write(path + '\n')

    print("=" * 50)
    print(f"Base patient paths: {len(base_patient_paths)}")
    print(f"Total AP subpaths found: {total_ap_count}")
    print(f"Successfully filtered out {remove_count} bad paths!")
    print(f"Good data numbers: {len(clean_patients_paths)}")
    print(f"Saved clean data list to: {clean_txt_path}")
    print("=" * 50)



#-------------------------------DataCleaning Area---------------------------------------

#clean_txt_path = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/DataCleaning/DataLists/Second_Good_data.txt'
#bad_txt_path = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/DataCleaning/Bad_Data/Truncation_check_bad_data.txt'
#CheckTruncation(clean_txt_path, bad_txt_path) 
#--------------------Truncation Check done. Found 143 abnormal dirs---------------------

#clean_txt_path = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/DataCleaning/DataLists/Second_Good_data.txt'
#bad_txt_path = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/DataCleaning/Bad_Data/middle_holes_check_bad_data.txt'
#check_middle_holes(clean_txt_path, bad_txt_path)
#--------------------Middle hole check done. Found 1 abnormal dirs----------------------

#clean_txt_path = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/DataCleaning/DataLists/Second_Good_data.txt'
#bad_txt_path = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/DataCleaning/Bad_Data/Second_Round_inspection/class_overlap_bad_data.txt'
#check_class_overlap(clean_txt_path, bad_txt_path)
#-------------------Class overlap check done. Found 63 abnormal dirs--------------------

txt_path = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/DataCleaning/DataLists/Second_Good_data.txt'
bad_txt_path = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/DataCleaning/Bad_Data/Final_Bad_Data.txt'
clean_txt_path = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/DataCleaning/Clean_Data.txt'
generate_clean_patient_list(txt_path, bad_txt_path, clean_txt_path)