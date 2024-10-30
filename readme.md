## 基于yolov8-obb的目标检测和分割的模型训练

**源码来自ultralytics的开源项目YOLOv8**

首先我们需要准备好具有标注好目标框的标注文件格式为txt，
标注文件的HBB标准格式为：

| class_id | x_center   | y_center   | width      | height     |
|----------|------------|------------|------------|------------|
| -        | 要求为0-1的归一化 | 要求为0-1的归一化 | 要求为0-1的归一化 | 要求为0-1的归一化 |

使用labelme标注好图片，导出为.json格式的标注数据，使用本项目中的
**[converter.py](transfomer.py)** 来将json文件中的标注转换为标准的HBB格式


如果需要obb训练，我们还要将这四个数据转换为四个角点x，y坐标，此事可在
**[transfomer.py](transfomer.py)** 进行转换

该python脚本完成对图片的旋转加噪，并且将HBB标注框转换为OBB。
请在[transfomer.py](transfomer.py)中填写你的数据集的文件夹，来完成。

```python
input_dir = 'C:/Users/chenj/Documents/GitHub/ultralytics/ultralytics/datasets'
output_dir = 'C:/Users/chenj/Documents/GitHub/ultralytics/ultralytics/Traindatasets-test'
```

### transformer.py只会对你的train数据集进行训练