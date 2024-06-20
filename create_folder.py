import os
import shutil
from pathlib import Path

# 創建資料夾並移動圖片
def organize_data(txt_file, base_dir):
    with open(txt_file, 'r') as f:
        for line in f:
            path, label = line.strip().split()
            label_dir = os.path.join(base_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            shutil.copy(path, label_dir)

# 設定路徑
train_txt = 'train.txt'
val_txt = 'val.txt'
test_txt = 'test.txt'

base_train_dir = 'train'
base_val_dir = 'val'
base_test_dir = 'test'

# 創建資料夾並整理資料
organize_data(train_txt, base_train_dir)
organize_data(val_txt, base_val_dir)
organize_data(test_txt, base_test_dir)

print("資料整理完成")