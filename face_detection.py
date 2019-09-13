import cv2
import matplotlib.pyplot as plt 

face_cascade = cv2.CascadeClassifier('/home/pi/opencv-3.4.3/data/haarcascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3,480) # set Width
cap.set(4,360) # set Height
while True:
	ret,img = cap.read()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
	faces = face_cascade.detectMultiScale(gray,1.3,5)
	
	for x,y,w,h in faces:
		gray_face = cv2.resize(gray[y:y+h,x:x+w],(100,100))
		cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),5)
	
	cv2.imshow('Face Detection',gray)
	plt.show()
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
