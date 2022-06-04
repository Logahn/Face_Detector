#!/usr/bin/env python3
import cv2
import  random 

trained_faces_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Choose image to detect
img = cv2.imread("images/multiple.png")

# Convert the data to greyscale
greyscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_faces_data.detectMultiScale(greyscaled_img)

# [[63 39 78 78]]

# Draw rectangles around faces
colors = ((256,0,0), (0,256,0), (0,0,256))

for (x, y, w, h) in face_coordinates:
    color = random.choice(colors)
    cv2.rectangle(img, (x, y),(x+w, y+h), color, 2)


# print(face_coordinates)


# cv2.imshow("Test data", img)
cv2.imshow("Test data", img)
cv2.waitKey()

#print("code completed")