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
print(name)

countValue = 0
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
                   (460, 20), m.fonts, 0.7, m.YELLOW, 2)
        RightEyePoint = PointList[36:42]
        LeftEyePoint = PointList[42:48]
        leftRatio, lTop, lBottom = m.blinkDetector(LeftEyePoint)
        rightRatio, rTop, rBottom = m.blinkDetector(RightEyePoint)

        blinkRatio = (leftRatio + rightRatio)/2
        cv.circle(frame, circleCenter, (int(blinkRatio*4.3)), m.CHOCOLATE, -1)
        cv.circle(frame, circleCenter, (int(blinkRatio*3.2)), m.CYAN, 2)
        cv.circle(frame, circleCenter, (int(blinkRatio*2)), m.GREEN, 3)

        if blinkRatio > 4:
            COUNTER += 1
            cv.putText(frame, f'Blink', (70, 50),
                       m.fonts, 0.8, m.LIGHT_BLUE, 2)
        else:
            if COUNTER > CLOSED_EYES_FRAME:
                TOTAL_BLINKS += 1
                COUNTER = 0
        cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (230, 17),
                   m.fonts, 0.5, m.ORANGE, 2)

        elapsed_time = time.time() - START_TIME
        mask, pos, color,eyeImage, bestMinT = m.EyeTracking(frame2, grayFrame, RightEyePoint, elapsed_time, bestMinT)
        maskleft, leftPos, leftColor,eyeImage, bestMinT = m.EyeTracking(
            frame2, grayFrame, LeftEyePoint, elapsed_time, bestMinT)
        mymask = np.bitwise_xor(mask, maskleft)

        countValue += 1

        # draw background as line where we put text.
        cv.line(frame, (30, 90), (100, 90), color[0], 30)
        cv.line(frame, (25, 50), (135, 50), m.WHITE, 30)
        cv.line(frame, (int(width-150), 50), (int(width-45), 50), m.WHITE, 30)
        cv.line(frame, (int(width-140), 90),
                (int(width-60), 90), leftColor[0], 30)

        # writing text on above line
        cv.putText(frame, f'{pos}', (35, 95), m.fonts, 0.6, color[1], 2)
        cv.putText(frame, f'{leftPos}', (int(width-140), 95),
                   m.fonts, 0.6, leftColor[1], 2)
        cv.putText(frame, f'Right Eye', (35, 55), m.fonts, 0.6, color[1], 2)
        cv.putText(frame, f'Left Eye', (int(width-145), 55),
                   m.fonts, 0.6, leftColor[1], 2)

    else:
        pass

    # Recoder.write(frame)
    # calculating the seconds
    SECONDS = time.time() - START_TIME
    # calculating the frame rate
    FPS = FRAME_COUNTER/SECONDS
    # print(FPS)
    # defining the key to Quite the Loop

    key = cv.waitKey(1)

    # if q is pressed on keyboard: quit
    if key == ord('q'):
        break
# closing the camera
camera.release()
# Recoder.release()
# closing  all the windows
cv.destroyAllWindows()