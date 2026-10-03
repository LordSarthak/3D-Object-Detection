from ultralytics import YOLO
import cv2
import cvzone
import math
 
cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)
#cap = cv2.VideoCapture("trail.mp4")  # For Video
 
model = YOLO("yolov8n.pt")  # Make sure this is trained on just ['Flatfoot', 'Normal']
 
classNames = ['Flatfoot', 'Normal']
myColor = (0, 0, 255)

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if cls < len(classNames):
                currentClass = classNames[cls]
                print(currentClass)

                if conf > 0.5:
                    if currentClass == 'Flatfoot':
                        myColor = (0, 0, 255)
                    elif currentClass == 'Normal':
                        myColor = (0, 255, 0)
                    else:
                        myColor = (255, 0, 0)

                    cvzone.putTextRect(
                        img,
                        f'{currentClass} {conf:.2f}',
                        (max(0, x1), max(35, y1)),
                        scale=1,
                        thickness=1,
                        colorB=myColor,
                        colorT=(255, 255, 255),
                        colorR=myColor,
                        offset=5
                    )
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)
            else:
                print(f"⚠️ Detected class ID {cls} is out of range for your classNames list.")

    cv2.imshow("Image", img)
    cv2.waitKey(1)