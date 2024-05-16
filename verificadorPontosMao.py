import cv2
import mediapipe as mp

# pegando o video da camera
video = cv2.VideoCapture(0)

# variavel para verificar a mao
hand = mp.solutions.hands

# variavel para fazer a detecção da mão dentro do video
Hand = hand.Hands(max_num_hands=1)

# variavel para desenhar o contorno da mão
mpDraw = mp.solutions.drawing_utils

# utilizando a camera
while True:
    # lendo o video
    check, img = video.read()

    # conversão da imagem para RGB
    frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # processando a imagem
    results = Hand.process(frameRGB)

    # identificando os pontos na mão
    handsPoints = results.multi_hand_landmarks

    # extraindo as dimensões da imagem
    h, w, _ = img.shape

    if handsPoints:
        # for para retornar as coordenadas de cada ponto da mão
        for points in handsPoints:
            # print(points)

            # desenhando o contorno da mão
            mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS)

            # printando os pontos
            for id, cord in enumerate(points.landmark):

                # desenhando os pontos da mão
                cx, cy = int(cord.x * w), int(cord.y * h)

                # # printando cada pontos da mão
                cv2.putText(img, str(id), (cx, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


                # printando a coordenada dos pontos
                # print(pontos)

    cv2.imshow("Imagem", img)
    cv2.waitKey(1)