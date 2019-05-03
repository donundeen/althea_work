import numpy as np
import cv2
import time
from picamera.array import PiRGBArray
from picamera import PiCamera

history_length = 400
nmixtures = 5
backgroundRatio = 0.7
noiseSigma = 0

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
 
# allow the camera to warmup
time.sleep(0.1)

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history_length, nmixtures, backgroundRatio, noiseSigma)


for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    fgmask = fgbg.apply(image)




    cv2.imshow('frame',fgmask)
    # show the frame
    #cv2.imshow("Frame2", image)
    key = cv2.waitKey(1) & 0xFF
    
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break



rawCapture.release()
cv2.destroyAllWindows()
