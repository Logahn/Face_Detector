#!/usr/bin/env python3
from tkinter.ttk import Frame
import cv2
import numpy as np
from PIL import Image
import os
import  random

trained_faces_data = cv2.CascadeClassifier("cascade.xml")
#* Choose image to detect
img1 = cv2.imread("images/data_test1.jpg")
img2 = cv2.imread("images/data_test2.jpeg")
img3 = cv2.imread("images/multiple_test.png")
#* Convert the data to greyscale
greyscaled_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
greyscaled_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
greyscaled_img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
 #* Detect faces
masked_coordinates1 = trained_faces_data.detectMultiScale(greyscaled_img1)
masked_coordinates2 = trained_faces_data.detectMultiScale(greyscaled_img2)
masked_coordinates3 = trained_faces_data.detectMultiScale(greyscaled_img3)
#* Draw rectangles around faces
colors = ((256,0,0), (0,256,0), (0,0,256))

for (x, y, w, h) in masked_coordinates1:
    color = random.choice(colors)
    cv2.rectangle(img1, (x, y),(x+w, y+h), color, 2)
    
for (x, y, w, h) in masked_coordinates2:
    color = random.choice(colors)
    cv2.rectangle(img2, (x, y),(x+w, y+h), color, 2)
    
for (x, y, w, h) in masked_coordinates3:
    color = random.choice(colors)
    cv2.rectangle(img3, (x, y),(x+w, y+h), color, 2)

cv2.imshow("Masked face", img1)
cv2.imshow("Unmasked face", img2)
cv2.imshow("Multiple faces", img3)
cv2.waitKey()
cv2.destroyAllWindows()