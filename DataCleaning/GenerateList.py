import os

target_path = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/nature/Dataset/internal_xxx'
outputDir = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/DataCleaning/DataLists'

all_patients = []
for patient_name in sorted(os.listdir(target_path)):
    all_patients.append((target_path, patient_name))

with open(os.path.join(outputDir, 'internal_xxx_list.txt'), 'w') as f:
    for folder, name in all_patients:
        patient_dir = os.path.join(folder, name)
        f.write(f"{patient_dir}\n")

print("Done")