import cv2
import numpy as np
import face_recognition
from PIL import Image,ImageDraw

font = cv2.FONT_HERSHEY_SIMPLEX

imgDurgesh = face_recognition.load_image_file('/home/pi/Desktop/FR/images/durgesh.PNG')
imgDurgesh_encoding = face_recognition.face_encodings(imgDurgesh)[0]

imgAlban = face_recognition.load_image_file('/home/pi/Desktop/FR/images/alban.jpg')
imgAlban_encoding = face_recognition.face_encodings(imgAlban)[0]

imgPrashant = face_recognition.load_image_file('/home/pi/Desktop/FR/images/Prashant.JPG')
imgPrashant_encoding = face_recognition.face_encodings(imgPrashant)[0]

imgAbhishek = face_recognition.load_image_file('/home/pi/Desktop/FR/images/Abhishek.JPG')
imgAbhishek_encoding = face_recognition.face_encodings(imgAbhishek)[0]

imgAnkita = face_recognition.load_image_file('/home/pi/Desktop/FR/images/Ankita.JPG')
imgAnkita_encoding = face_recognition.face_encodings(imgAnkita)[0]

known_face_encodings = [imgDurgesh_encoding,imgAlban_encoding,imgPrashant_encoding,imgAbhishek_encoding,imgAnkita_encoding]
known_face_names = ['Durgesh','Alban','Prashant','Abhishek','Ankita']

cam = cv2.VideoCapture(2)

while True:
    ret, img =cam.read()
    # testImg = face_recognition.load_image_file(img)
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img,face_locations)
    
    for (top,right,bottom,left),face_encoding in zip(face_locations,face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings,face_encoding)
        name = 'Unknown'
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        cv2.rectangle(img,(left,top),(right,bottom),(255,0,0),5)
        cv2.putText(img, name, (left+2,top+23), font, 1, (255,255,255), 3)
        cv2.imshow('camera',img)

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()


