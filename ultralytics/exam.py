from ultralytics import YOLO

model = YOLO("C:/Users/chenj/Documents/GitHub/ultralytics/runs/obb/train6/weights/best.pt")


results = model("./rotateimages/159.jpg",
                save=True)

print(results)