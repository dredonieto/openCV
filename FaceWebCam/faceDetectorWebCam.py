import numpy as np
import cv2
import logging
import sys

class FaceDetector:

    def __init__(self, name="image"):
        self.cap_ = cv2.VideoCapture(1)
        self.face_cascade_ = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.number_ = 0
        self.name_ = name
        self.training_ = False
        self.size_ = 0
        logging.basicConfig(level=logging.INFO)

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
            for rect in faces:
                (x,y,w,h) = rect
                cv2.rectangle(frame,(x-self.size_, y-self.size_),(x+w+self.size_ ,y+h+self.size_),(255,0,0),2)
                
            key = cv2.waitKey(1)
            # Press s to save and image of your detected face     
            if key == ord('s'):
                crop_img = frame[y-self.size_:y+h+self.size_, x-self.size_:x+w+self.size_]
                self.number_ += 1
                self.saveImage(crop_img, self.number_)
                logging.info('Image saved succesfully')
            elif key == ord('q'):
                self.stop()
            elif key == ord('b'):
                self.size_ = self.size_ + 2
                logging.info('Doing rectangle bigger')
            elif key == ord('m'):
                self.size_ = self.size_ - 2
                logging.info('Doing rectangle smaller')
            # Display the resulting frame
            cv2.imshow('frame',frame)
            

    #Save image of your face only for training
    def saveImage(self, img, number):
        name = self.name_ + str(number) + '.jpg'
        cv2.imwrite(name, img)

    def stop(self):
        # When everything done, release the capture
        self.cap_.release()
        cv2.destroyAllWindows()
        


if __name__ == "__main__":
    if len(sys.argv) != 2:
        detector = FaceDetector()
        detector.start()
    else:
        detector = FaceDetector(sys.argv[1])
        detector.start()
    