import os
import random
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
import shutil  # 用于复制文件

# Define paths
base_path = Path('datasets/water_surface')
images_dir = base_path / 'JPEGImages'
annotations_dir = base_path / 'labels'
train_images_dir = base_path / 'train/images'
train_labels_dir = base_path / 'train/labels'
val_images_dir = base_path / 'val/images'
val_labels_dir = base_path / 'val/labels'
test_images_dir = base_path / 'test/images'
test_labels_dir = base_path / 'test/labels'

# Create directories if they don't exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

# List all image files
image_files = list(images_dir.glob('*.jpg'))  # 根据需要调整扩展名

# Shuffle the image files
random.shuffle(image_files)

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Calculate split indices
num_images = len(image_files)
train_split = int(num_images * train_ratio)
val_split = int(num_images * (train_ratio + val_ratio))

# Split images and labels
train_images = image_files[:train_split]
val_images = image_files[train_split:val_split]
test_images = image_files[val_split:]

def copy_files(source_images, dest_images_dir, dest_labels_dir):
    for img_file in source_images:
        label_file = annotations_dir / (img_file.stem + '.txt')
        if label_file.exists():
            try:
                # 使用 shutil.copy2 复制文件而不是创建符号链接
                shutil.copy2(img_file, dest_images_dir / img_file.name)
                shutil.copy2(label_file, dest_labels_dir / label_file.name)
                print(f"Copied {img_file.name} and its label.")
            except Exception as e:
                print(f"Error copying {img_file.name}: {e}")

copy_files(train_images, train_images_dir, train_labels_dir)
copy_files(val_images, val_images_dir, val_labels_dir)
copy_files(test_images, test_images_dir, test_labels_dir)

print("Dataset splitting completed.")