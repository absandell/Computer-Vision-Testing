import cv2
import mediapipe as mp
import time

# 0. WRIST
# 1. THUMB_CMC
# 2. THUMB_MCP
# 3. THUMB_IP
# 4. THUMB_TIP
# 5. INDEX_FINGER_MCP
# 6. INDEX_FINGER_PIP
# 7. INDEX_FINGER_DIP
# 8. INDEX_FINGER_TIP
# 9. MIDDLE_FINGER_MCP
# 10. MIDDLE_FINGER_PIP
# 11. MIDDLE_FINGER_DIP
# 12. MIDDLE_FINGER_TIP
# 13. RING_FINGER_MCP
# 14. RING_FINGER_PIP
# 15. RING_FINGER_DIP
# 16. RING_FINGER_TIP
# 17. PINKY_MCP
# 18. PINKY_PIP
# 19. PINKY_DIP
# 20. PINKY_TIP

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
                #print(id, landmark)
                h, w, c = img.shape
                cx, cy, = int(landmark.x*w), int(landmark.y*h)
                print(id, cx, cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 20, (0, 255, 255), cv2.FILLED) # Draws circle on specific  reference point id
            mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS) # Draw on original image (each hand)

    # FPS Measure
    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2, )

    cv2.imshow("Image", img)
    cv2.waitKey(1)

