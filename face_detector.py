#!/usr/bin/env python3
import cv2
import  random 

trained_faces_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# #* Choose image to detect
# img = cv2.imread("images/multiple.png")

# #* Convert the data to greyscale
# greyscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# #* Detect faces
# face_coordinates = trained_faces_data.detectMultiScale(greyscaled_img)

# #* Draw rectangles around faces
# colors = ((256,0,0), (0,256,0), (0,0,256))

# for (x, y, w, h) in face_coordinates:
#     color = random.choice(colors)
#     cv2.rectangle(img, (x, y),(x+w, y+h), color, 2)

# cv2.imshow("Test data", img)
# cv2.waitKey()


#* Capture video from webcam
webcam = cv2.VideoCapture(0)

#* Iterate forever over frames
while True:

    #* Read current frame
    successful_frame_read, frame = webcam.read() 

    #* Convert the data to greyscale
    greyscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # #* Detect faces
    face_coordinates = trained_faces_data.detectMultiScale(greyscaled_frame)

     #* Draw rectangles around faces
    colors = ((256,0,0), (0,256,0), (0,0,256))

    for (x, y, w, h) in face_coordinates:
        color = random.choice(colors)
        cv2.rectangle(frame, (x, y),(x+w, y+h), color, 2)

    cv2.imshow("Test data", frame)
    cv2.waitKey(2)




#print("code completed")