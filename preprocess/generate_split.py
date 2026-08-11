import random
import os

test_path = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/Dataset_AP/positive_sample/positive_samples.txt'

# Read all non-empty lines into a list
with open(test_path, 'r') as f:
    all_patients = [line.strip() for line in f if line.strip()]

print(f"Total patients read: {len(all_patients)}")


random.seed(16) #Salute to my favorite Formula 1 driver, Charles LECLERC
                #For his wonderful driving in SilverStone and Spa
                #Not today, Not tomorrow, But one day, He will be WDC!!!
random.shuffle(all_patients)

split_idx = int(len(all_patients)*0.8)
train_patients = all_patients[:split_idx]
test_patients = all_patients[split_idx:]

outputDir = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/Dataset_AP/positive_sample'

with open(os.path.join(outputDir, 'train.txt'), 'w') as f:
    for patient_dir in train_patients:
        f.write(f"{patient_dir}\n")

with open(os.path.join(outputDir, 'test.txt'), 'w') as f:
    for patient_dir in test_patients:
        f.write(f"{patient_dir}\n")