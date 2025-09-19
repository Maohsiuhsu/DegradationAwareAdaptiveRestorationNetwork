import os
import shutil
import random

def split_and_copy(A_folder, B_folder, train_folder, val_folder, train_ratio=0.9):
    # 如果訓練集和驗證集資料夾不存在，創建它們
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # 瀏覽 A 和 B 資料夾中的檔案，假設兩個資料夾中有相同檔名的檔案
    A_files = os.listdir(A_folder)
    B_files = os.listdir(B_folder)
    
    # 確保 A 和 B 資料夾中的檔案名稱對應
    common_files = set(f.replace('_real_A', '') for f in A_files).intersection(
        set(f.replace('_fake_B', '') for f in B_files))
    
    # 將檔案隨機打亂
    common_files = list(common_files)
    random.shuffle(common_files)
    
    # 計算訓練集和驗證集的分割點
    train_size = int(len(common_files) * train_ratio)
    
    # 分割訓練集和驗證集
    train_files = common_files[:train_size]
    val_files = common_files[train_size:]
    
    # 設置 A 和 B 資料夾的訓練集和驗證集資料夾
    A_train_folder = os.path.join(train_folder, 'normal')
    B_train_folder = os.path.join(train_folder, 'wet')
    A_val_folder = os.path.join(val_folder, 'normal')
    B_val_folder = os.path.join(val_folder, 'wet')

    # 創建對應的資料夾
    os.makedirs(A_train_folder, exist_ok=True)
    os.makedirs(B_train_folder, exist_ok=True)
    os.makedirs(A_val_folder, exist_ok=True)
    os.makedirs(B_val_folder, exist_ok=True)
    
    # 複製檔案到訓練集
    for file in train_files:
        file_name = file.split('.')[0]
        file_type = file.split('.')[1]
        A_file_path = os.path.join(A_folder, file_name + '_real_A' + '.' + file_type)
        B_file_path = os.path.join(B_folder, file_name + '_fake_B' + '.' + file_type)
        shutil.copy(A_file_path, os.path.join(A_train_folder, os.path.basename(A_file_path)))
        shutil.copy(B_file_path, os.path.join(B_train_folder, os.path.basename(B_file_path)))
    
    # 複製檔案到驗證集
    for file in val_files:
        file_name = file.split('.')[0]
        file_type = file.split('.')[1]
        A_file_path = os.path.join(A_folder, file_name + '_real_A' + '.' + file_type)
        B_file_path = os.path.join(B_folder, file_name + '_fake_B' + '.' + file_type)
        shutil.copy(A_file_path, os.path.join(A_val_folder, os.path.basename(A_file_path)))
        shutil.copy(B_file_path, os.path.join(B_val_folder, os.path.basename(B_file_path)))

# 使用範例
A_folder = '/home/ubuntul40/Howard/Dataset/88x88_new/normal'
B_folder = '/home/ubuntul40/Howard/Dataset/88x88_new/wet'
train_folder = '/home/ubuntul40/Howard/Dataset/airdunet/train'
val_folder = '/home/ubuntul40/Howard/Dataset/airdunet/val'

split_and_copy(A_folder, B_folder, train_folder, val_folder)
