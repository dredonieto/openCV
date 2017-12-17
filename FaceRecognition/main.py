import numpy as np
import cv2
import sys


subjects = ["" , "David", "Mama", "Elena"]

# Function to draw rectangle on image
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Function to draw text
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: main.py 'ClassiferName' ")
        sys.exit()

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(sys.argv[1])
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.open(0)
    while(cap.isOpened()):
        ret, frame = cap.read()
        #cv2.imshow('frame', frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Extract the faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for rect in faces:
            (x,y,w,h) = rect
            face_person = gray[y:y+w, x:x+h]
            # Predict every face in the picture
            label, confidence = face_recognizer.predict(face_person)
            label_text = subjects[label]
            draw_rectangle(frame, rect)
            draw_text(frame, label_text, rect[0], rect[1]-5)
        cv2.imshow('Face recognizer', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    