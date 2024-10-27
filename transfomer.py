import os
import albumentations as A
import numpy as np
import shutil
import random
import cv2

# 输入输出路径
input_dir = 'C:/Users/chenj/Documents/GitHub/ultralytics/ultralytics/datasets'
output_dir = 'C:/Users/chenj/Documents/GitHub/ultralytics/ultralytics/Traindatasets-test'

# 创建数据增强方法
transform = A.Compose([
    A.GaussianBlur(p=0.3),
    A.RandomBrightnessContrast(p=0.2),
    A.MotionBlur(blur_limit=5, p=0.2),
    A.ImageCompression(quality_lower=70, p=0.3),
    A.CLAHE(p=0.2),
    A.ToGray(p=1.0),
    A.Perspective(scale=(0.02, 0.05), p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def add_background_noise(image):
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def calculate_obb_corners(x_center, y_center, width, height, angle, w, h):
    x_center, y_center = x_center * w, y_center * h
    width, height = width * w, height * h
    corners = np.array([
        [x_center - width / 2, y_center - height / 2],
        [x_center + width / 2, y_center - height / 2],
        [x_center + width / 2, y_center + height / 2],
        [x_center - width / 2, y_center + height / 2]
    ])
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    ones = np.ones(shape=(len(corners), 1))
    corners_ones = np.hstack([corners, ones])
    rotated_corners = M.dot(corners_ones.T).T
    normalized_corners = rotated_corners / np.array([w, h])
    return normalized_corners

def rotate_image_and_bbox_to_obb(image, bboxes, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    rotated_bboxes = [calculate_obb_corners(*bbox, angle, w, h).flatten() for bbox in bboxes]
    return rotated_image, rotated_bboxes

def save_image_and_label(image, bboxes, image_output_path, label_output_path, class_labels):
    cv2.imwrite(image_output_path, image)
    with open(label_output_path, 'w') as f:
        for corners, class_id in zip(bboxes, class_labels):
            label = [class_id] + corners.tolist()
            f.write(' '.join(map(str, label)) + '\n')

def process_image_folder(image_dir, label_dir, output_image_dir, output_label_dir, augment=True):
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name.replace('.jpg', '.txt'))
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            continue
        height, width, _ = image.shape
        bboxes, class_labels = [], []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())
                bboxes.append([x_center, y_center, bbox_width, bbox_height])
                class_labels.append(int(class_id))
        num_augmentations = 10 if augment else 1
        for i in range(num_augmentations):
            processed_image = add_background_noise(image) if augment else image.copy()
            rotate_angle = random.uniform(-20, 20) if augment else random.uniform(-10, 10)
            if augment:
                transformed = transform(image=processed_image, bboxes=bboxes, class_labels=class_labels)
                processed_image, transformed_bboxes = transformed['image'], transformed['bboxes']
            else:
                transformed_bboxes = bboxes
            rotated_image, rotated_bboxes = rotate_image_and_bbox_to_obb(processed_image, transformed_bboxes, rotate_angle)
            output_image_name = f"{image_name.replace('.jpg', '')}_aug_{i}.jpg"
            output_label_name = output_image_name.replace('.jpg', '.txt')
            save_image_and_label(rotated_image, rotated_bboxes,
                                 os.path.join(output_image_dir, output_image_name),
                                 os.path.join(output_label_dir, output_label_name),
                                 class_labels)

def setup_output_dirs():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

def main():
    setup_output_dirs()
    process_image_folder(
        image_dir=os.path.join(input_dir, 'images', 'train'),
        label_dir=os.path.join(input_dir, 'labels', 'train'),
        output_image_dir=os.path.join(output_dir, 'images', 'train'),
        output_label_dir=os.path.join(output_dir, 'labels', 'train'),
        augment=True
    )
    process_image_folder(
        image_dir=os.path.join(input_dir, 'images', 'val'),
        label_dir=os.path.join(input_dir, 'labels', 'val'),
        output_image_dir=os.path.join(output_dir, 'images', 'val'),
        output_label_dir=os.path.join(output_dir, 'labels', 'val'),
        augment=False
    )
    print("训练和验证数据集的增强与转换已完成，结果保存到:", output_dir)

if __name__ == "__main__":
    main()
