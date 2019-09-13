import cv2
import os
import numpy as np

cam = cv2.VideoCapture(1)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('/home/pi/opencv-3.4.3/data/haarcascades/haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('/home/pi/opencv-3.4.3/data/haarcascades/haarcascade_eye.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <Enter> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while(True):
    ret, img = cam.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_detector.detectMultiScale(gray,1.3,10)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        
        
        count += 1
       
        # Save the captured image into the datasets folder
        cv2.imwrite("/home/pi/Desktop/Face Recognition Project/Training Faces/durgesh/User." + str(face_id) + '.' + str(count) + ".jpg",gray[y:y+h,x:x+h])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 100: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
