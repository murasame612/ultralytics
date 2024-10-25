from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data="C:/Users/chenj/Documents/GitHub/yolov10/ultralytics/train-data/config.yaml",
            epochs=100)

results = model("C:/Users/chenj/Documents/GitHub/yolov10/ultralytics/train-data/test/images/148.jpg",
                save=True)