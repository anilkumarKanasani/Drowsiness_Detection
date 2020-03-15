################### Importing all librires ###########################
import cv2
import numpy as np
import dlib
from math import hypot
from threading import Thread,Timer
import time
import vlc
p = vlc.MediaPlayer("file:///ambulanz.mp3")

################### Preparing initial level variables ###########################
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

################### User defined functions###########################
def start_sound() : 
    p.play()
    dur = p.get_length() / 1000
    time.sleep(dur)

def midpt (p1,p2): return (int((p1.x+p2.x)/2) , int((p1.y+p2.y)/2))

def display_pts(li,facial_landmarks,color):
    for i in li: cv2.circle(frame , 
                            (facial_landmarks.part(i).x , facial_landmarks.part(i).y) ,
                            1 , color , 2)

def get_blinking_ratio (eye_points , facial_landmarks) :
    left_pt = (facial_landmarks.part(eye_points[0]).x , facial_landmarks.part(eye_points[0]).y) 
    right_pt = (facial_landmarks.part(eye_points[3]).x , facial_landmarks.part(eye_points[3]).y) 
    top_pt = midpt (facial_landmarks.part(eye_points[1]) , facial_landmarks.part(eye_points[2]))
    btm_pt = midpt (facial_landmarks.part(eye_points[5]) , facial_landmarks.part(eye_points[4]))
    hr_ln = cv2.line(frame,left_pt , right_pt , (0,255,0) , 1)
    vr_ln = cv2.line(frame,top_pt , btm_pt , (0,255,0) , 1)
    hr_ln_len = hypot(left_pt[0]-right_pt[0] , 
                         left_pt[1]-right_pt[1] )
        
    vr_ln_len = hypot(top_pt[0]-btm_pt[0] , 
                         top_pt[1]-btm_pt[1] )
        
    ratio = hr_ln_len / (vr_ln_len+0.001)
    return ratio

################### Main function ###########################

while True:
    _ , frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces :
        landmarks = predictor(gray,face)
        lft_eye_ratio = get_blinking_ratio([36,37,38,39,40,41] , landmarks)
        rt_eye_ratio = get_blinking_ratio([42,43,44,45,46,47] , landmarks)
        bln_ratio = (lft_eye_ratio + rt_eye_ratio) /2
        
        if bln_ratio > 5 :
            count = count+1
            cv2.putText(frame, "EYES CLOSED" , (10,30) , font , 0.7 , (0,0,255),2)
            cv2.putText(frame, "COUNTER = " + str(count) , (10,60) , font , 0.7 , (0,0,255),2)
            display_pts(np.arange(36,48),landmarks,(0,0,255))
            t = Thread(target=start_sound)
            t.deamon = True
            if count >= 10: t.start()
            
        else : 
            cv2.putText(frame, "EYES OPEN" , (10,30) , font , 0.7 , (0,255,0),2)
            count = 0
            p.stop()
            display_pts(np.arange(36,48),landmarks,(0,255,0))
    
        
    cv2.imshow('Frame' , frame)
    
    key = cv2.waitKey(1)
    if key == ord('q') : break



cap.release()
cv2.destroyAllWindows()
