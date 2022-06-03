#!/usr/bin/env python3
import cv2

trained_faces_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Choose image to detect
img = cv2.imread("RDJ.jpg")

cv2.imshow("Test data", img)
cv2.waitKey()
# Convert the data to greyscale


#print("code completed")