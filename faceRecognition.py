#OpenCV module
import cv2
#os module for reading training data directories and paths
import os
#numpy to convert python lists to numpy arrays as it is needed by OpenCV face recognizers
import numpy as np

subjects = ["" , "David", "Mama"]

# Function to detect face using OpenCV
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if (len(faces) == 0):
        return None, None
    # Only one face
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

# Function to draw rectangle on image
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Function to draw text
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

# Function to predict
def predict(test_img):
    # Make a copy to avoid changing the original
    img = test_img.copy()
    face, rect = detect_face(img)
    label = face_recognizer.predict(face)
    label_text = subjects[label]
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)
    return img

print("Let's prepare the training data")
faces, labels = prepare_training_data("training_data")
print("Faces detected: ", len(faces))

# We are going to use LBPH recognizer
face_recognizer = cv2.face.createLBPHFaceRecognizer()
face_recognizer.train(faces, np.array(labels))

print("Now, let's predict some faces...")
# Load test image
test1 = cv2.imread("test_data/img1.jpg")
# Make the prediction
predict1 = predict(test1)
cv2.imshow(subjects[1], predict1)
cv2.waitKey(0)
cv2.destroyAllWindows()
