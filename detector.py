from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2
import numpy as np
import sys
import time
# np.set_printoptions(threshold=sys.maxsize)
import torch


print(torch.cuda.is_available())

def create_bar(height, width, color):

    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    return bar, (red, green, blue)




apple_detected = False
orange_detected = False

apple_counter = 0
orange_counter = 0

stop_detection = False


frame_counter = 0
prev_frame_time = 0
new_frame_time = 0
error_counter = 0

yolo_classes=["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

model = YOLO("yolov8m.pt")



cam = cv2.VideoCapture(2)

while True:
    ret_val, img = cam.read()

    if not ret_val:
        error_counter += 1
        if error_counter > 10:
            break
        continue


    print("Debug: frame :", frame_counter)
    print("DEBUG: FPS: ",)

    frame_counter += 1
    frame = cv2.resize(img, (640, 480))

    results = model(frame)

    result = results[0]

    print(result)

    bboxes = np.array(result.boxes.xyxy.cpu(),dtype="int")
    classes = np.array(result.boxes.cls.cpu(),dtype="int")
    confidences = np.array(result.boxes.conf.cpu())

    # print("bboxes: " , bboxes , " Done")

    for cls, bbox,conf in zip(classes,bboxes,confidences):
        (x,y,x2,y2) = bbox

        cv2.rectangle(frame,(x,y),(x2,y2),(0,0,255),2)

        clsconf = str(cls) + ". "+ yolo_classes[cls]+" "+ str(conf)

        cv2.putText(frame,str(clsconf) ,(x,y-5),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)

        if cls == 47:
            apple_coordinates = [x,y,x2,y2]


    # print(classes)

    if 47 in classes:
        apple_counter = apple_counter + 1
        print("Debug: Apple counter: ", apple_counter)
        if (apple_counter == 5):
            print("APPLE FOUND!!")
            x,y,x2,y2 = apple_coordinates
            # print(apple_coordinates)
            crop_image = frame[y:y2, x:x2]
            cv2.imshow("Crop_img",crop_image)

            height, width, _ = np.shape(crop_image)
            data = np.reshape(crop_image,(height*width,3))
            data = np.float32(data)

            number_clusters = 1
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
            flags = cv2.KMEANS_RANDOM_CENTERS
            compactness,labels,centers = cv2.kmeans(data,number_clusters,None,criteria,10,flags)
            print(centers)

            if(centers[0,2] > 200 and centers[0,1] < 150):
                print("RED APPLE!!!!!!!")
            else:
                print("Yellow Apple!!!!!!")

            stop_detection = True
    else:
        apple_counter = 0
    if 49 in classes:
        orange_counter = orange_counter + 1
        print("Debug: Orange counter: ", orange_counter)
        if (orange_counter == 5):
            print("Orange FOUND!!")
            stop_detection = True
    else:
        orange_counter = 0


    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    cv2.putText(frame, fps, (7, 70),cv2.FONT_HERSHEY_PLAIN, 3, (100, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("img",frame)


    if cv2.waitKey(1) == 27 or stop_detection:
        stop_detection = False
        # time.sleep(100)
        error_counter = 0
        break  # esc to quit

cam.release()
cv2.destroyAllWindows()

