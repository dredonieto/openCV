import numpy as np
import yaml
import cv2

class FaceDetector:

    def __init__(self):
        self.cap_ = cv2.VideoCapture(1)
        self.face_cascade_ = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.number_ = 0
        self.name_ = "default"
        self.training_ = False

    def setUp(self, doc):
        self.name_ = doc["name"]
        self.training_ = doc["training"]

    def start(self):
        if not self.cap_.isOpened():
            self.cap_.open(0)
        while(self.cap_.isOpened()):
            # Capture frame-by-frame
            ret, frame = self.cap_.read()

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #Extract the faces
            faces = self.face_cascade_.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                #TODO: delete rectangle from the image before saved
                if self.training_:
                    crop_img = frame[y:y+h, x:x+w]
                    self.number_ += 1
                    self.saveImage(crop_img, self.number_)

            # Display the resulting frame
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()

    #Save image of your face only for training
    def saveImage(self, img, number):
        name = self.name_ + str(number) + '.jpg'
        cv2.imwrite(name, img)

    def stop(self):
        # When everything done, release the capture
        cap_.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    with open('parameters.yaml','r') as f:
        doc = yaml.load(f)
    detector = FaceDetector()
    detector.setUp(doc)
    detector.start()