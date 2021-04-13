import cv2
import os
import numpy as np
import detectface as fr

#read the test file put the test file path here
test=cv2.imread('D:/NIGAM/PROGRAMMING/Projects/face rec/test.jpg')
faces_detected,gray_img=fr.facedet(test)
print("Faces deteced:",faces_detected)

for (x,y,w,h) in faces_detected:
   cv2.rectangle(test,(x,y),(x+w,y+h),(255,0,0),thickness=5)
cv2.imshow("Face Detected",resizes_img)
cv2.waitKey(0)
cv2.destroyAllWindows

faces,faceID=fr.lables_for_training('D:/NIGAM/Programming/face rec/training')
face_recognizer=fr.tarin_classifier(faces,faceID)

name={0:"person1",1:"person2"}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    lable,confidence=face_recognizer.predict(roi_gray)
    print("Confidence:",confidence)
    print("label:",lable)
    fr.draw_rect(test,face)
    predict_name=name[lable]
    fr.put_text(test,predict_name,x,y)

resized_img=cv2.resize(test,(720,640))
cv2.imshow("Face Detected",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows

        
