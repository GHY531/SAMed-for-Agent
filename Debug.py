import numpy as np

# 加载 .npy 文件
label = np.load('/home/bml/storage/mnt/v-3f30eb9261b04a32/org/nature/Dataset/internal_2024/陈素芬/CT00819397/AP/label_result.npy')

# 获取所有出现的唯一类别
unique_classes = np.unique(label)

print("包含的类别列表:", unique_classes)
print("类别总数（含背景）:", len(unique_classes))