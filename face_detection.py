import cv2
import os
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt


main_path = r'E:\SanChain\deeplearning\tensorflow\face\eg1'
model = YOLO("yolov8n-face.pt")


def face_detect(img):
    detection = model(img)
    boxes = detection[0].boxes

    return boxes

def draw_box(boxes,img):

    for box in boxes:
        x = int(box.xyxy.tolist()[0][0])
        y = int(box.xyxy.tolist()[0][1])
        w = int(box.xyxy.tolist()[0][2])
        h = int(box.xyxy.tolist()[0][3])
           
        cv2.rectangle(img,(x,y),(w,h),(50,200,129),2)
    return img




for folder in os.listdir(main_path):
    files = os.path.join(main_path,folder)
    photo_path = os.path.join(files, 'photo') 
    if not os.path.exists(photo_path):
        os.makedirs(photo_path)

    for file in os.listdir(files):
        if file.endswith('.mp4'):
            video_file = os.path.join(files,file)
            print(video_file)
            count = 0
            cap = cv2.VideoCapture(video_file)

            if not cap.isOpened():
                print("error opening video file")
                exit()
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(f"FPS : {fps}, width:{frame_width} , height : {frame_height}")

            while True:
                ret , frame = cap.read()

                if not ret:
                    break

                frame = np.array(frame,"uint8")
                boxes = face_detect(frame)
                for box in boxes:
                    x = int(box.xyxy.tolist()[0][0])
                    y = int(box.xyxy.tolist()[0][1])
                    w = int(box.xyxy.tolist()[0][2])
                    h = int(box.xyxy.tolist()[0][3])
                    box_area = (w-x) * (h-y)
                    print("Bounding Box Area:", box_area)
                if box_area > 30000:
                    count = count + 1
                    frame_ = draw_box(boxes,frame)
                    cv2.imshow(f"Video {folder}",frame_)
                    img = frame[y:h,x:w]
                    crop_img = cv2.resize(img,(224,224))
                    
                    img_name = os.path.join(photo_path,f"{folder}_{count}.jpg")
                    cv2.imwrite(img_name,crop_img)


                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            
cap.release()
cv2.destoryAllWindows()
