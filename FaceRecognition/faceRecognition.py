import cv2
import os
import numpy as np
import sys


subjects = ["" , "David", "Mama", "Elena"]

# Function to detect face using OpenCV
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if (len(faces) == 0):
        return None, None
    # Only one face because our training set has only one face
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

# Function to prepare training data
def prepare_training_data(folder_path):
    dirs = os.listdir(folder_path)
    faces = []
    labels = []
    for dir_name in dirs:
        # Our training directories start with the letter t
        if not dir_name.startswith("t"):
            continue;
        # The directories containing training data are named 'tX' 
        # where X is a number
        label = int(dir_name.replace("t", ""))
        dir_path = folder_path + "/" + dir_name
        image_names = os.listdir(dir_path)
        for image_name in image_names:
            # ignore sytem files
            if image_name.startswith("."):
                continue;
            image_path = dir_path + "/" + image_name
            image = cv2.imread(image_path)
            # Detect the face
            face, rect = detect_face(image)
            if face is not None:
                faces.append(face)
                labels.append(label)
    return faces, labels


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: faceRecognition.py 'ClassiferName' ")
        sys.exit()
        
    print("Let's prepare the training data")
    faces, labels = prepare_training_data("training_data")
    print("Faces detected: ", len(faces))

    # We are going to use LBPH recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))
    name = sys.argv[1] + ".xml"
    face_recognizer.save(name)
    print('classifier saved succesfully')
