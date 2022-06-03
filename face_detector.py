#!/usr/bin/env python3
import cv2

trained_faces_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Choose image to detect
img = cv2.imread("RDJ.jpg")

# Convert the data to greyscale
greyscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



cv2.imshow("Test data", img)
cv2.imshow("Test data", greyscaled_img)
cv2.waitKey()

#print("code completed")