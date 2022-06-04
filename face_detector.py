#!/usr/bin/env python3
import cv2
import  random

trained_faces_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Choose image to detect
img = cv2.imread("RDJ2.jpg")

# Convert the data to greyscale
greyscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_faces_data.detectMultiScale(greyscaled_img)

# [[63 39 78 78]]

# Draw rectangles around faces
# cv2.rectangle(img, (63, 39), (63+78, 39+78), (0, 255,0), 2)

b = random.randint(0,255)
g = random.randint(0,255)
r = random.randint(0,255)


for (x, y, w, h) in face_coordinates:
    color = (r,b,g)
    cv2.rectangle(img, (x, y),(x+w, y+h), color, 2)


# print(face_coordinates)


# cv2.imshow("Test data", img)
cv2.imshow("Test data", img)
cv2.waitKey()

#print("code completed")