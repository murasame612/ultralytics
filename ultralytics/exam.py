from ultralytics import YOLO

model = YOLO("C:/Users/chenj/Documents/GitHub/ultralytics/runs/obb/train7/weights/best.pt")


results = model("./rotateimages/159.jpg",
                save=False)

for result in results:
    obb = result.obb
    result.show()
    print(obb)