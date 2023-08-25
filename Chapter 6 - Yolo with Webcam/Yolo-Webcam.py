from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("../Yolo-Weights/yolov8n.pt")

classnames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
              "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            #Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            w, h = x2-x1, y2-y1


            #Confidence
            conf = math.ceil((box.conf[0]*100))/100

            #Class name
            cls = int(box.cls[0])
            if  cls == 1:
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'{classnames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
            #else:
                #cvzone.putTextRect(img, f'{cls} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)