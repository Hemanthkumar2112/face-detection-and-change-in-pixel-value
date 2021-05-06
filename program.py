import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
class FaceDetect:   
   
    def __init__(self, path ,haarcascade_path):   
        self.path = path 
        self.haarcascade_path = haarcascade_path
        self.org = (50, 50) 
        self.fontScale = 1
        self.color = (255, 0, 0) 
        self.thickness = 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX 
        
    def detect_face(self):
        face_cascade = cv2.CascadeClassifier(self.haarcascade_path)
        img = cv2.imread(self.path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
 
            ints = [x,y,w,h]
            string_ints = [str(int) for int in ints]
            str_of_ints = ",".join(string_ints)
            rotate_img =cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE) 
            cv2.putText(rotate_img, str_of_ints , self.org, self.font, self.fontScale,  self.color, self.thickness, cv2.LINE_AA, False) 
            cv2.imwrite('bounding box on the Face.jpg', rotate_img)
            x = rotate_img.shape[:2]
            for i in tqdm(range(x[0])):
                for j in range(x[1]):
                    y = rotate_img[i,j]
                    if y[0] in range(200,255):
                        rotate_img[i,j]= 0     
                    if y[1] in range(200,255):
                        rotate_img[i,j] = 255
                    if y[2] in range(200,255):
                        rotate_img[i,j] = 0
            cv2.imshow('rotate_image_with_bound', rotate_img)
            cv2.imwrite('Replace all pixels.jpg', rotate_img)
            cv2.waitKey(0) 
        
p = FaceDetect('/home/hemanth/Desktop/55279.jpg','/home/hemanth/Desktop/haarcascade_frontalface_default.xml')   
p.detect_face()
