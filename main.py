import cv2 as cv
import numpy as np
import module as m
import time
import json
import matplotlib.pyplot as plt

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

image = cv.imread('images/9quadrantes.jpeg')

# Obtém as dimensões da tela
screen_width = 1920 # Coloque a largura da sua tela
screen_height = 1080 # Coloque a altura da sua tela
# Redimensiona a imagem para o tamanho da tela
image = cv.resize(image, (screen_width, screen_height))

# Mostra a imagem em tela cheia
cv.namedWindow('Imagem', cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty('Imagem', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
cv.imshow('Imagem', image)
cv.waitKey(1)
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

with open('cache/informationEye.json') as f:
    dados = json.load(f)

#maior_x = max(dado["X"] for dado in dados if isinstance(dado, dict) and "X" in dado)
#maior_y = max(dado["Y"] for dado in dados if isinstance(dado, dict) and "Y" in dado)
#menor_x = min(dado["X"] for dado in dados if isinstance(dado, dict) and "X" in dado)
#menor_y = min(dado["Y"] for dado in dados if isinstance(dado, dict) and "Y" in dado)

x = [dado["X"] for dado in dados  if isinstance(dado, dict)]
y = [dado["Y"] for dado in dados  if isinstance(dado, dict)]



# Crie um dicionário para armazenar a contagem de cada ponto
pontos = {}
for i in range(len(x)):
    ponto = (x[i], y[i])
    if ponto in pontos:
        pontos[ponto] += 1
    else:
        pontos[ponto] = 1

# Separe os pontos únicos e repetidos
pontos_unicos = [ponto for ponto, count in pontos.items() if count == 1]
pontos_repetidos = [ponto for ponto, count in pontos.items() if count > 1]
tamanho_repetidos = [pontos[ponto] * 50 for ponto in pontos_repetidos]  # Tamanho proporcional ao número de repetições

# Plote os pontos únicos
plt.scatter([ponto[0] for ponto in pontos_unicos], [ponto[1] for ponto in pontos_unicos], color='blue', label='Pontos únicos')

# Plote os pontos repetidos com tamanhos maiores
plt.scatter([ponto[0] for ponto in pontos_repetidos], [ponto[1] for ponto in pontos_repetidos], s=tamanho_repetidos, color='blue', label='Pontos repetidos')

# Adicione legendas e rótulos
plt.legend()
plt.xlabel('Valores de X')
plt.ylabel('Valores de Y')
plt.title('Gráfico de Dispersão com Pontos Repetidos Destacados')
plt.grid(True)

# Exiba o gráfico
plt.show()