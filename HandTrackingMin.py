import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils # What we use to draw on hands

previousTime = 0
currentTime = 0

while True:
    success, img = cap.read() # Reads video input
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Defines RGB version of image to use in processing
    results = hands.process(imgRGB) # Processes image
    #print(results.multi_hand_landmarks) # Provides coordinates of any hands on screen

    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks: # Extract info from each hand
            for id, landmark in enumerate(handLandmarks.landmark):
                print(id, landmark)
            mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS) # Draw on original image (each hand)

    # FPS Measure
    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2, )

    cv2.imshow("Image", img)
    cv2.waitKey(1)

