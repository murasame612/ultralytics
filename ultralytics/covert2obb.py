import os
import numpy as np

def convert_yolo_to_obb(yolo_labels):
    obb_labels = []
    for label in yolo_labels:
        class_id, x_center, y_center, width, height = label
        x_center, y_center, width, height = float(x_center), float(y_center), float(width), float(height)

        # 计算四个角点的坐标
        x1, y1 = x_center - width / 2, y_center - height / 2
        x2, y2 = x_center + width / 2, y_center - height / 2
        x3, y3 = x_center + width / 2, y_center + height / 2
        x4, y4 = x_center - width / 2, y_center + height / 2

        obb_labels.append([class_id, x1, y1, x2, y2, x3, y3, x4, y4])
    return obb_labels

def process_labels(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with open(input_path, 'r') as file:
                yolo_labels = [list(map(float, line.strip().split())) for line in file.readlines()]

            obb_labels = convert_yolo_to_obb(yolo_labels)

            with open(output_path, 'w') as file:
                for label in obb_labels:
                    file.write(' '.join(map(str, label)) + '\n')

# 示例使用
input_folder = 'datasets/labels/val_original'
output_folder = 'datasets-converted/labels/val_original'
process_labels(input_folder, output_folder)
