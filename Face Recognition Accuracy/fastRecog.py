import face_recognition
import cv2
import numpy as np
import pickle

video_capture = cv2.VideoCapture(1)
#all_face_encodings = {}

#imgDurgesh = face_recognition.load_image_file('/home/pi/Desktop/FR/images/durgesh.PNG')
#all_face_encodings["durgesh"] = face_recognition.face_encodings(imgDurgesh)[0]

#imgAlban = face_recognition.load_image_file('/home/pi/Desktop/FR/images/alban.jpg')
#all_face_encodings["alban"] = face_recognition.face_encodings(imgAlban)[0]

#imgPrashant = face_recognition.load_image_file('/home/pi/Desktop/FR/images/Prashant.JPG')
#all_face_encodings["prashant"] = face_recognition.face_encodings(imgPrashant)[0]

#imgAbhishek = face_recognition.load_image_file('/home/pi/Desktop/FR/images/Abhishek.JPG')
#all_face_encodings["abhishek"] = face_recognition.face_encodings(imgAbhishek)[0]

#imgAnkita = face_recognition.load_image_file('/home/pi/Desktop/FR/images/Ankita.JPG')
#all_face_encodings["ankita"] = face_recognition.face_encodings(imgAnkita)[0]

#with open('dataset_faces.dat', 'wb') as f:
#    pickle.dump(all_face_encodings, f)
    
# Load face encodings
with open('dataset_faces.dat', 'rb') as f:
	all_face_encodings = pickle.load(f)

# Grab the list of names and the list of encodings
known_face_names = list(all_face_encodings.keys())
known_face_encodings = np.array(list(all_face_encodings.values()))

# Initialize some variables
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        #face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                 first_match_index = matches.index(True)
                 name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            # face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            #best_match_index = np.argmin(face_distances)
            #if matches[best_match_index]:
            #    name = known_face_names[best_match_index]

            #face_names.append(name)

    process_this_frame = not process_this_frame


 
    for (top, right, bottom, left), name in zip(face_locations, known_face_names):
       
        top *= 5
        right *= 5
        bottom *= 5
        left *= 5

       
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

     
        cv2.rectangle(frame, (left, bottom +25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom + 20), font, 1.0, (255, 255, 255), 1)

    
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()

