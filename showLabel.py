import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np

def read_bboxes_from_txt(label_path):
    """
    从文本文件中读取标注框
    :param label_path: 标签文件路径
    :return: 标注框列表，每个标注框包含四个角点的坐标（归一化到0-1范围）
    """
    bboxes = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])
            corners = parts[1:]
            bboxes.append(corners)
    return bboxes

def visualize_bboxes(image_path, bboxes):
    """
    可视化四角点标注框
    :param image_path: 图像文件路径
    :param bboxes: 标注框列表，每个标注框包含四个角点的坐标
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for bbox in bboxes:
        corners = np.array(bbox).reshape(4, 2) * np.array([image.shape[1], image.shape[0]])
        polygon = patches.Polygon(corners, closed=True, edgecolor='g', linewidth=2, fill=False)
        ax.add_patch(polygon)

    plt.show()

# 示例使用
image_path = 'C:/Users/chenj/Documents/GitHub/ultralytics/ultralytics/TrainDatasets/images/train/149_aug_1.jpg'
label_path = 'C:/Users/chenj/Documents/GitHub/ultralytics/ultralytics/TrainDatasets/labels/train/149_aug_1.txt'

# 从标签文件中读取标注框
bboxes = read_bboxes_from_txt(label_path)

# 可视化标注框
visualize_bboxes(image_path, bboxes)



# 示例使用


