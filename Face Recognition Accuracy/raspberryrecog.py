import cv2
import numpy as np
import face_recognition
from PIL import Image,ImageDraw
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)

TRIG = 7
ECHO = 12
LEDB = 15

GPIO.setup(LEDB,GPIO.OUT)
GPIO.setup(TRIG,GPIO.OUT)
GPIO.output(TRIG,0)

GPIO.setup(ECHO,GPIO.IN)

time.sleep(0.1)


font = cv2.FONT_HERSHEY_SIMPLEX

imgDurgesh = face_recognition.load_image_file('/home/pi/Desktop/user img/durgesh_thakur.JPG')
imgDurgesh_encoding = face_recognition.face_encodings(imgDurgesh)[0]

imgAlban = face_recognition.load_image_file('/home/pi/Desktop/user img/alban_sheikh.JPG')
imgAlban_encoding = face_recognition.face_encodings(imgAlban)[0]

imgHariom = face_recognition.load_image_file('/home/pi/Desktop/user img/hariom.JPG')
imgHariom_encoding = face_recognition.face_encodings(imgHariom)[0]

imgChandan = face_recognition.load_image_file('/home/pi/Desktop/user img/chandan.jpg')
imgChandan_encoding = face_recognition.face_encodings(imgChandan)[0]

imgDeepika = face_recognition.load_image_file('/home/pi/Desktop/user img/deepika.JPG')
imgDeepika_encoding = face_recognition.face_encodings(imgDeepika)[0]

known_face_encodings = [imgDurgesh_encoding,imgAlban_encoding,imgHariom_encoding,imgChandan_encoding,imgDeepika_encoding]
known_face_names = ['Durgesh','Alban','Hariom','Chandan','Deepika']

cam = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    
    GPIO.output(TRIG,1)
    time.sleep(0.00001)
    GPIO.output(TRIG,0)

    while GPIO.input(ECHO)==0:
        pass
        
    start = time.time()
    
    while GPIO.input(ECHO)==1:
        pass
        
    stop = time.time()
    
    time.sleep(0.1)
    # Grab a single frame of video
    ret, frame = cam.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if (stop-start)*17000 <=10:
        GPIO.output(LEDB,1)
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
        

    else :
        GPIO.output(LEDB,0)
        face_names = []


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        print(name)
        
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
