# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 22:05:59 2018

@author: Horse
"""

#import os
#import pandas as pd 
import csv
import cv2



cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('Classifiers/face.xml')
i = 0
offset = 50
ID = input('Enter your ID : ')
name = input('Enter your Name : ')
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        i=i+1
        cv2.imwrite("DataSet/"+name+"-"+ID +'.'+ str(i) + ".jpg", gray[y-offset:y+h+offset,x-offset:x+w+offset])
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        cv2.imshow('im',im[y-offset:y+h+offset,x-offset:x+w+offset])
        cv2.waitKey(100)
    if i>20:
        cam.release()
        cv2.destroyAllWindows()
        break
with open("users-data.csv" , "a") as csvfile:
    filewriter = csv.writer(csvfile, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
    #------
    filewriter.writerow([name, ID, 'F:\Face Recognation\Dataset'])