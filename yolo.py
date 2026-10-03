from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load pre-trained model
model.train(data="datayolo/images/data.yaml", epochs=50, imgsz=640)
