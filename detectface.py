import cv2
import os
import numpy as np



def facedet(test):
    gray_img=cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
    #mention the path that you want to put xml file 
    face_haar_cascade=cv2.CascadeClassifier('D:/NIGAM/PROGRAMMING/Projects/face rec/haarcascade_frontalface_default.xml')
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.39,minNeighbors=11)
    return faces,gray_img

#labeling the data

def lables_for_training(directory):
     faces=[]
     faceID=[]

     for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                 print("Skipping sys file")
                 continue

            id=os.path.basename(path)
            img_path=os.path.join(path,filename)
            print("img_path",img_path)
            print("id:",id)
            test=cv2.imread(img_path)
            if test is None:
                 print("Image not loaded properly")
                 continue
            faces_rect,gray_img=facedet(test)
            if len(faces_rect)!=1:
                 continue #one 1 person's image
            (x,y,w,h)=faces_rect[0]
            roi_gray=gray_img[y:y+w,x:x+h]
            faces.append(roi_gray)
            faceID.append(int(id))
     return faces,faceID

def tarin_classifier(faces,faceID):
     face_recognizer=cv2.face.LBPHFaceRecognizer_create()
     face_recognizer.train(faces,np.array(faceID))
     return face_recognizer

def draw_rect(test,face):
     (x,y,w,h)=face
     cv2.rectangle(test,(x,y),(x+w,y+h),(255,0,0),thickness=5)

def put_text(test,text,x,y):
     cv2.putText(test,text,(x,y),cv2.FONT_HERSHEY_COMPLEX,5,(255,0,0),5)
 
