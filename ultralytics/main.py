from ultralytics import YOLO

model = YOLO("../yolov8n-obb.pt")

# results = model("https://ultralytics.com/images/bus.jpg",show=True,save=True)

model.train(data="C:/Users/chenj/Documents/GitHub/ultralytics/ultralytics/config.yaml",
            task ='obb',epochs=100)