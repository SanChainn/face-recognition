import cv2
import os
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications import ResNet50
#from tensorflow.keras.applications import ResNet101
from tensorflow.keras.models import load_model


#main_path = r'E:\SanChain\deeplearning\tensorflow\face\eg1'
main_path = r'E:\SanChain\deeplearning\tensorflow\face\gg'
model = YOLO("yolov8n-face.pt")

#loaded_model = load_model("three_face_resnet152.h5")
loaded_model = load_model("three_face_resnet50.h5")
#loaded_model = load_model("three_face_resnet101.h5")



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

def draw_box_label(boxes,img,label):

    for box in boxes:
        x = int(box.xyxy.tolist()[0][0])
        y = int(box.xyxy.tolist()[0][1])
        w = int(box.xyxy.tolist()[0][2])
        h = int(box.xyxy.tolist()[0][3])     

        cv2.rectangle(img,(x,y),(w,h),(50,200,129),2)


        # Add label text
        label = label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_thickness = 3
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_x = x + 5  # Adjust the x-coordinate to position the text to the right of the box
        text_y = y - 10  # Adjust the y-coordinate to position the text above the box
        cv2.putText(img, label, (text_x, text_y), font, font_scale, (50, 200, 129), font_thickness)


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

            f_cnt = cv2.CAP_PROP_FRAME_COUNT
            numFrames = int(cap.get(f_cnt))
            print(f"Numframes : {numFrames}")

            skip_frames = 2
            frame_count = 0


            if not cap.isOpened():
                print("error opening video file")
                exit()
            
            # fps = cap.get(cv2.CAP_PROP_FPS)
            # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # print(f"FPS : {fps}, width:{frame_width} , height : {frame_height}")

            for i in range(0,numFrames, skip_frames):
            
                ret , frame = cap.read()

                if not ret:
                    break

                frame_count += 1
                
                # if frame_count % skip_frames == 0:

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
                    
                    img = frame[y:h,x:w]
                    crop_img = cv2.resize(img,(224,224))
                    
                    img_array = image.img_to_array(crop_img)
                    img_array = np.expand_dims(img_array, axis = 0)
                    img_array = img_array / 255.0

                    # prediction 
                    predictions = loaded_model.predict(img_array)
                    predicted_class = np.argmax(predictions)
                    # map predict class
                    class_labels = ['cillian murphy','kyaw ze yar htun','ronaldo']
                    predicted_label = class_labels[predicted_class]
                    
                    frame_ = draw_box_label(boxes,frame,predicted_label)
                    cv2.imshow(f"Video {folder}",frame_)
                        
                        # img_name = os.path.join(photo_path,f"{folder}_{count}.jpg")
                        # cv2.imwrite(img_name,crop_img)


                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            
cap.release()
cv2.destoryAllWindows()
