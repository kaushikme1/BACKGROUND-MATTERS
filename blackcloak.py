import cv2
import time
import numpy as np
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('result.avi',fourcc,20.0,(640,480))
cap = cv2.VideoCapture(0)
time.sleep(3)
count = 0
background = 0
for i in range(60):
    ret, background = cap.read()
background = np.flip(background, axis = 1)
while(cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    count += 1
    img = np.flip(img, axis = 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    I_black = np.array([0,100,25])
    u_black = np.array([0,255,255])
    mask1 = cv2.inRange(hsv, u_black,I_black)
    I_black = np.array([170,100,65])
    u_black = np.array([100,255,255])
    mask2 = cv2.inRange(hsv, u_black,I_black)
    mask1 = mask1 + mask2
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
    mask2 = cv2.bitwise_not(mask1) 
    res1 = cv2.bitwise_and(img,img,mask = mask2)
    res2 = cv2.bitwise_and(background,background,mask = mask1)
    finalOutput = cv2.addWeighted(res1,1,res2,1,0)
    out.write(finalOutput)
    cv2.imshow("BLACK WAS HERE  ",finalOutput)
    cv2.waitKey(1)
cap.release()
out.release()
cv2.destroyAllWindows()       