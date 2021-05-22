import cv2 
import numpy as np

url = 'http://192.168.1.69:8080/video'
cap = cv2.VideoCapture(url)
#cap = cv2.VideoCapture(0)

backSub = cv2.createBackgroundSubtractorMOG2()

last_frame = None

def processImage(frame):
    global last_frame
    
    """gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)

    frame = cv2.bitwise_and(frame,frame,mask = ~thresh)
    fgMask = backSub.apply(frame)

    frame = fgMask"""
    
    """frame = cv2.resize(frame,(400,300))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if last_frame is None:
        last_frame = frame
        return frame
    
    diff = cv2.absdiff(last_frame,frame)
    last_frame = frame"""

    return frame

while(True):
    ret, frame = cap.read()
    if frame is not None:
        processed_frame = processImage(frame)
        cv2.imshow('frame',processed_frame)
    q = cv2.waitKey(1)
    if q == ord("q"):
        break
cv2.destroyAllWindows()