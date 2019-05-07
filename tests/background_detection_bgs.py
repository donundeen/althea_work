import numpy as np
import cv2
import bgs
import time
from picamera.array import PiRGBArray
from picamera import PiCamera

history_length = 800 #200
nmixtures = 5 #5
backgroundRatio = 0.3    #0.7
noiseSigma = 0

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
#camera.resolution = (64, 64)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

#better ones
algorithm = bgs.AdaptiveSelectiveBackgroundLearning()
#algorithm.threshold = 3
#   alphaLearn(0.05), alphaDetection(0.05), learningFrames(-1), counter(0), minVal(0.0), maxVal(1.0),  threshold(15)
#algorithm = bgs.AdaptiveBackgroundLearning()
#algorithm = bgs.FrameDifference()
#algorithm = bgs.StaticFrameDifference()

#algorithm = bgs.LOBSTER()
#algorithm = bgs.DPEigenbackground()

 
# allow the camera to warmupq
time.sleep(0.1)

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history = history_length, nmixtures = nmixtures, backgroundRatio = backgroundRatio, noiseSigma = noiseSigma)


for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    fgmask = algorithm.apply(image)
    fgmask = np.fliplr(fgmask)


    # Threshold.
    # Set values equal to or above 220 to 0.
    # Set values below 220 to 255.
     
    th, im_th = cv2.threshold(fgmask, 220, 255, cv2.THRESH_BINARY_INV);
     
    # Copy the thresholded image.
    im_floodfill = im_th.copy()
     
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
     
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
     
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
     
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv


    #fgmask = im_out  




    

    fgmask_crop = fgmask[0:480 , 80:560]

    small = cv2.resize(fgmask_crop, (0,0), fx=0.133, fy=0.133) 

    cv2.imshow('MOG',fgmask_crop)
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
