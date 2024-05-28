import cv2 as cv
import numpy as np
import dlib
import math
#from screeninfo import get_monitors
import json
import time
import os
# variables
fonts = cv.FONT_HERSHEY_COMPLEX

# face detector object
detectFace = dlib.get_frontal_face_detector()
# landmarks detector
predictor = dlib.shape_predictor(
    "Predictor/shape_predictor_68_face_landmarks.dat")

def midpoint(pts1, pts2):
    x, y = pts1
    x1, y1 = pts2
    xOut = int((x + x1)/2)
    yOut = int((y1 + y)/2)
    return (xOut, yOut)


def eucaldainDistance(pts1, pts2):
    x, y = pts1
    x1, y1 = pts2
    eucaldainDist = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)

    return eucaldainDist

# creating face detector function


def faceDetector(image, gray, Draw=False):
    cordFace1 = (0, 0)
    cordFace2 = (0, 0)
    # getting faces from face detector
    faces = detectFace(gray)
    face = None
    # looping through All the face detected.
    for face in faces:
        # getting coordinates of face.
        cordFace1 = (face.left(), face.top())
        cordFace2 = (face.right(), face.bottom())

        if Draw == True:
            cv.rectangle(image, cordFace1, cordFace2, (0, 255, 0), 2)
    return face


def faceLandmakDetector(image, gray, face, Draw=False):
    # calling the landmarks predictor
    landmarks = predictor(gray, face)
    landmarks.parts()
    pointList = [(p.x, p.y) for p in landmarks.parts()]
    if Draw == True:
        for point in pointList:
            cv.circle(image, point, 3, (0, 69, 255), 1)
    return pointList

# Blink detector function.


def blinkDetector(eyePoints):
    top = eyePoints[1:3]
    bottom = eyePoints[4:6]
    # finding the mid point of above points
    topMid = midpoint(top[0], top[1])
    bottomMid = midpoint(bottom[0], bottom[1])
    # getting the actual width and height eyes using eucaldainDistance function
    VerticalDistance = eucaldainDistance(topMid, bottomMid)
    HorizontalDistance = eucaldainDistance(eyePoints[0], eyePoints[3])

    blinkRatio = (HorizontalDistance/VerticalDistance)
    return blinkRatio, topMid, bottomMid

# Eyes Tracking function.


def EyeTracking(image, gray, eyePoints, time, bestMin, startTime):
    # getting dimensions of image
    dim = gray.shape
    # creating mask .
    mask = np.zeros(dim, dtype=np.uint8)

    # converting eyePoints into Numpy arrays.
    PollyPoints = np.array(eyePoints, dtype=np.int32)
    # Filling the Eyes portion with WHITE color.
    cv.fillPoly(mask, [PollyPoints], 255)
    # Writing gray image where color is White  in the mask using Bitwise and operator.
    eyeImage = cv.bitwise_and(gray, gray, mask=mask)

    # getting the max and min points of eye inorder to crop the eyes from Eye image .

    maxX = (max(eyePoints, key=lambda item: item[0]))[0]#acha a cordenada com maior valor de x e extrai apenas o valor de x dela
    minX = (min(eyePoints, key=lambda item: item[0]))[0]
    maxY = (max(eyePoints, key=lambda item: item[1]))[1]
    minY = (min(eyePoints, key=lambda item: item[1]))[1]

    # other then eye area will black, making it white
    eyeImage[mask == 0] = 255

    # cropping the eye form eyeImage.
    cropedEye = eyeImage[minY:maxY, minX:maxX]
    teste = gray[minY:maxY, minX:maxX]
    cv.imshow('cropedEye', teste)
    cv.GaussianBlur(src=teste,ksize=(3,3),sigmaX=0)
    PupilaTest(teste)
    cv.GaussianBlur(src=cropedEye,ksize=(7,7),sigmaX=0)
    #_, thresholdEye = cv.threshold(cropedEye,58 , 255, cv.THRESH_BINARY)
   # if time <= 5:
    #    bestMin,_ = bestthreshold(cropedEye, bestMin)
    #_, bestThresholdEye = cv.threshold(cropedEye, bestMin, 255, cv.THRESH_BINARY)
    #cv.imshow('BestThresh', bestThresholdEye)

    #centery = centerEyey(bestThresholdEye)
    #centerx = centerEyex(bestThresholdEye)
    #if time > 5:
     #   cordenates = dictFormatCenter(centerx, centery, cropedEye.shape[0], cropedEye.shape[1], startTime)
        #if cordenates != -1:
            #jsonFormat(cordenates)
   # center = np.ones((bestThresholdEye.shape), dtype=int).astype(np.uint8) * 255
   # print(centerx, centery)
    #center[centerx, centery] = 0
   # center = cv.cvtColor(center, cv.COLOR_GRAY2BGR)

    #cv.imshow('centerEye', center)

    return mask, cropedEye, bestMin

def dictFormat( X, Y, width, height, starttime):
    dictonary = {}
    quadrante1 = (X > 0 and X < width/3) and (Y > 0 and Y < height/3)
    quadrante2 = (X > width/3 and X < 2 * (width/3)) and (Y > 0 and Y < height/3)
    quadrante3 = (X > 2 * (width/3) and X < width) and (Y > 0 and Y < height/3)
    quadrante4 = (X > 0 and X < width/3) and (Y > height/3 and Y < 2*(height/3))
    quadrante5 = (X > width/3 and X < 2 * (width/3)) and (Y > height/3 and Y < 2*(height/3))
    quadrante6 = (X > 2 * (width/3) and X < width) and (Y > height/3 and Y < 2*(height/3))
    quadrante7 = (X > 0 and X < width/3) and (Y > 2*(height/3) and Y < height)
    quadrante8 = (X > width/3 and X < 2 * (width/3)) and (Y > 2*(height/3) and Y < height)
    quadrante9 = (X > 2 * (width/3) and X < width) and (Y > 2*(height/3) and Y < height)

    if quadrante1:
        dictonary = dict(
            horizontal = 'esquerda',
            vertical = 'cima'
        )

    elif quadrante2:
        dictonary = dict(
            horizontal = 'centro',
            vertical = 'cima'
        )

    elif quadrante3:
        dictonary = dict(
            horizontal = 'direita',
            vertical = 'cima'
        )
    elif quadrante4:
        dictonary = dict(
            horizontal = 'esquerda',
            vertical = 'centro'
        )
    elif quadrante5:
        dictonary = dict(
            horizontal = 'centro',
            vertical = 'centro'
        )
    elif quadrante6:
        dictonary = dict(
            horizontal = 'direita',
            vertical = 'centro'
        )
    elif quadrante7:
        dictonary = dict(
            horizontal = 'esquerda',
            vertical = 'baixo'
        )
    elif quadrante8:
        dictonary = dict(
            horizontal = 'centro',
            vertical = 'baixo'
        )
    elif quadrante9:
        dictonary = dict(
            horizontal = 'direita',
            vertical = 'baixo'
        )
    else:
        return -1
    dictonary['instante'] = time.time() - starttime
    print(dictonary)
    return dictonary

def dictFormatCenter( X, Y, width, height, starttime):
    #dictonary = {}
    if X != None and Y != None:
        dictonary = dict(
                X = float(X),
                Y = float(Y),
                instante = time.time()- starttime,
                )
        print(dictonary)
        return dictonary
    else:
        return -1


def jsonFormat(information):
    if os.path.exists('cache/informationEye.json'):
        with open('cache/informationEye.json', 'r') as f:
            data = json.load(f)
    else:
        data = []
    data.append(information)
    with open("cache/informationEye.json", 'w') as f:
        json.dump(data, f, indent=2)

def bestthreshold(crop, bestMin = 0):
    maxX = int(np.array(crop).shape[0])
    maxY = int(np.array(crop).shape[1])
    centerX = int(maxX/2)
    centerY = int(maxY/2)    
    minX = 0

    leftCenter = crop[maxX-1,centerY]
    rightCenter = crop[minX, centerY]

    bestMaxThreshold = max(leftCenter, rightCenter)
    bestMinThreshold = crop[centerX, centerY]
    bestMinThreshold = max(bestMinThreshold, bestMin)
    
    return bestMinThreshold, bestMaxThreshold

def centerEyex(croppedEye):
    non_zero_counts = np.count_nonzero(croppedEye, axis=1)
    min_line_index = np.argmin(non_zero_counts)
    return min_line_index

def centerEyey(croppedEye):
    non_zero_counts = np.count_nonzero(croppedEye, axis=0)
    min_column_index = np.argmin(non_zero_counts)
    return min_column_index

def PupilaTest(img):
    copia = img.copy()
    #Extrai as cores entre o intervalo BGR definido
    #mask = cv.inRange(img, (0, 0, 0), (60, 60, 60))
    # slice no preto
    #imask = mask > 0
    #preto = np.zeros_like(img, np.uint8)
   # preto[imask] = img[imask]

    #preto = cv.cvtColor(preto, cv.COLOR_BGR2GRAY)
    #cv.imshow('Preto', img)

    # detecção de círculos
    circles = cv.HoughCircles(copia, cv.HOUGH_GRADIENT, 2, 5,
                            param1=30, param2=30, minRadius=4, maxRadius=100)
    print(circles)

    #param do Grab Cut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # pelo menos um círculo encontrado
    if circles is not None:
        # converte para int
        print('passei por aqui')
        # loop nas coordenadas (x, y) e raio dos círculos encontrados
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
        # draw the outer circle
            cv.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv.circle(img,(i[0],i[1]),2,(0,0,255),3)
            cv.imshow('detected circles',img)
        