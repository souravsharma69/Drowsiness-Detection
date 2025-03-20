import tensorflow as tf
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
model=tf.keras.models.load_model("eyes.keras")
cascade=cv2.CascadeClassifier("haarcascade_eye (1).xml")
import cv2
cap=cv2.VideoCapture(0)
while cap.isOpened():
    _,img=cap.read()
    faces=cascade.detectMultiScale(img,scaleFactor=1.10,minNeighbors=4)
    for x,y,w,h in faces:
        face=img[y:y+h,x:x+w]
        cv2.imwrite("face.jpg",face)
        face=image.load_img("face.jpg",target_size=(150,150))
        face=image.img_to_array(face)
        face=np.expand_dims(face,axis=0)
        ans=model.predict(face)
        if ans>0.5:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(img,"o",(x,y),2,2,(0,255,0))
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
            cv2.putText(img,"c",(x,y),2,2,(255,0,0))
    cv2.imshow("You are under Sourav's Survillance",img)
    if cv2.waitKey(1)==97:
         break
cap.release()
cv2.destroyAllWindows()
