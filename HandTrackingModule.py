import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5,  trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils # What we use to draw on hands

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Defines RGB version of image to use in processing
        results = self.hands.process(imgRGB) # Processes image
        #print(results.multi_hand_landmarks) # Provides coordinates of any hands on screen

        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks: # Extract info from each hand
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS) # Draw on original image (each hand)

                #for id, landmark in enumerate(handLandmarks.landmark):
                    #print(id, landmark)
                #    h, w, c = img.shape
                #    cx, cy, = int(landmark.x*w), int(landmark.y*h)
                #    print(id, cx, cy)
                #    if id == 0:
                #        cv2.circle(img, (cx, cy), 15, (0, 255, 255), cv2.FILLED) # Draws circle on specific  reference point id


def main():
    previousTime = 0
    currentTime = 0
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()  # Reads video input

        # FPS Measure
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2, )

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()