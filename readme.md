## 基于yolov8-obb的目标检测和分割的模型训练

*源码来自ultralytics的开源项目YOLOv8*

### 1.数据标注和预处理

首先我们需要准备好具有标注好目标框的标注文件格式为txt，
标注文件的HBB标准格式为：

| class_id | x_center   | y_center   | width      | height     |
|----------|------------|------------|------------|------------|
| -        | 要求为0-1的归一化 | 要求为0-1的归一化 | 要求为0-1的归一化 | 要求为0-1的归一化 |

使用labelme标注好图片，导出为.json格式的标注数据，使用本项目中的
**[converter.py](transfomer.py)** 来将json文件中的标注转换为标准的HBB格式

**目录结构：**
<pre>
|_datasets
    |_images
        |_train
        |_val
    |_labels
        |_train
        |_val

</pre>


如果需要obb训练，我们还要将这四个数据转换为四个角点x，y坐标，此事可在
**[transfomer.py](transfomer.py)** 进行转换

该python脚本完成对图片的旋转加噪，并且将HBB标注框转换为OBB。
请在transfomer.py中填写你的数据集的文件夹，来完成这个任务。

```python
input_dir = 'C:/Users/chenj/Documents/GitHub/ultralytics/ultralytics/datasets'
output_dir = 'C:/Users/chenj/Documents/GitHub/ultralytics/ultralytics/Traindatasets-test'
```

### transformer.py会对你的train和val数据集进行转换
对数据集val进行普通的旋转，对数据集train经行随机的加噪，并且把HBB转换为
obb格式。

### 2.填写yaml配置文件

新建一个[ultralytics/config.yaml](ultralytics/config.yaml)文件，里面填写了验证集
和训练集图片的存储位置

    train: C:/Users/chenj/Documents/GitHub/ultralytics/ultralytics/datasets-converted/images/train
    val: C:/Users/chenj/Documents/GitHub/ultralytics/ultralytics/datasets-converted/images/val
    nc: 1
    obb: true
    names: ['0']

train指的是训练集图片的位置，val指明了测试集图片的位置
nc=标签数，names存储了标签的名字

| 变量名     | 含义        |
|---------|-----------|
| train   | 训练集图片的位置  |
| val     | 测试集图片的位置  |
| nc      | 标签数       |
| names   | 标签名列表     |

### 3.训练模型

在[main.py](ultralytics/main.py)中训练模型
```python
from ultralytics import YOLO
model = YOLO("../yolov8n-obb.pt")#预加载obb模型
model.train(data="ultralytics/databoost.yaml",
            task ='obb',epochs=100)
```

### 4.得到目标检测模型

训练完的模型会保存到/runs

![example](examples/val_batch0_pred.jpg)

