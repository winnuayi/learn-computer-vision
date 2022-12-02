# https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html

# $ pip uninstall opencv-python opencv-python-headless
# $ pip install opencv-contrib-python

import numpy as np
import cv2 as cv

PATH = "rtmp://192.168.100.95:1935/awesome/awesome"
cap = cv.VideoCapture(PATH)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 100)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 100)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv.destroyAllWindows()