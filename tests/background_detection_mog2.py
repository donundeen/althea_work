import numpy as np
import cv2
import time
from picamera.array import PiRGBArray
from picamera import PiCamera

history_length = 800 #200
nmixtures = 10 #5
backgroundRatio = 0.3 #0.7
noiseSigma = 0

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
#camera.resolution = (64, 64)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
 
# allow the camera to warmup
time.sleep(0.1)

fgbg = cv2.createBackgroundSubtractorMOG2(history=800, varThreshold=4, detectShadows=False)


for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array


    fgmask = fgbg.apply(image)
    fgmask = np.fliplr(fgmask)

    fgmask_crop = fgmask[0:480 , 80:560]

    small = cv2.resize(fgmask_crop, (0,0), fx=0.133, fy=0.133) 

    cv2.imshow('MOG2',fgmask_crop)
    # show the frame
    #cv2.imshow("Frame2", image)
    key = cv2.waitKey(1) & 0xFF
    
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break



#camera.release()
#rawCapture.release()
cv2.destroyAllWindows()
