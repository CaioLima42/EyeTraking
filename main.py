import cv2 as cv
import numpy as np
import module as m
import time

# Variables
COUNTER = 0
TOTAL_BLINKS = 0
CLOSED_EYES_FRAME = 3
cameraID = 0
videoPath = "Video/Your Eyes Independently_Trim5.mp4"
# variables for frame rate.
FRAME_COUNTER = 0
START_TIME = time.time()
FPS = 0

# creating camera object
camera = cv.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
f = camera.get(cv.CAP_PROP_FPS)
width = camera.get(cv.CAP_PROP_FRAME_WIDTH)
height = camera.get(cv.CAP_PROP_FRAME_HEIGHT)
print(width, height, f)
fileName = videoPath.split('/')[1]
name = fileName.split('.')[0]
bestMinT = 0

while True:
    FRAME_COUNTER += 1
    # getting frame from camera
    ret, frame = camera.read()
    ret, frame2 = camera.read()
    if ret == False:
        break

    # converção da imagem para escala de cinza.
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    height, width = grayFrame.shape
    circleCenter = (int(width/2), 50)
    # calling the face detector funciton
    face = m.faceDetector(frame, grayFrame)
    if face is not None:
        # calling landmarks detector funciton.
        PointList = m.faceLandmakDetector(frame, grayFrame, face)

        cv.putText(frame, f'FPS: {round(FPS,1)}',
                   (460, 20), m.fonts, 0.7, (0, 247, 255), 2)
        RightEyePoint = PointList[36:42]
        LeftEyePoint = PointList[42:48]
        leftRatio, lTop, lBottom = m.blinkDetector(LeftEyePoint)
        rightRatio, rTop, rBottom = m.blinkDetector(RightEyePoint)

        blinkRatio = (leftRatio + rightRatio)/2

        if blinkRatio > 4:
            COUNTER += 1
            cv.putText(frame, f'Blink', (70, 50),
                       m.fonts, 0.8, (255, 9, 2), 2)
        else:
            if COUNTER > CLOSED_EYES_FRAME:
                TOTAL_BLINKS += 1
                COUNTER = 0
        cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (230, 17),
                   m.fonts, 0.5, (0, 69, 255), 2)

        elapsed_time = time.time() - START_TIME
        mask, eyeImage, bestMinT = m.EyeTracking(frame2, grayFrame, RightEyePoint, elapsed_time, bestMinT, START_TIME)
        maskleft,eyeImage, bestMinT = m.EyeTracking(frame2, grayFrame, LeftEyePoint, elapsed_time, bestMinT, START_TIME)
        mymask = np.bitwise_xor(mask, maskleft)

        cv.imshow('frame', frame)
    else:
        pass

    SECONDS = time.time() - START_TIME
    # calculating the frame rate
    FPS = FRAME_COUNTER/SECONDS

    key = cv.waitKey(1)

    # if q is pressed on keyboard: quit
    if key == ord('q'):
        break
# closing the camera
camera.release()
# Recoder.release()
# closing  all the windows
cv.destroyAllWindows()