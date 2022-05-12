import cv2
import mediapipe as mp
import time

# 0. nose
# 1. left_eye_inner
# 2. left_eye
# 3. left_eye_outer
# 4. right_eye_inner
# 5. right_eye
# 6. right_eye_outer
# 7. left_ear
# 8. right_ear
# 9. mouth_left
# 10. mouth_right
# 11. left_shoulder
# 12. right_shoulder
# 13. left_elbow
# 14. right_elbow
# 15. left_wrist
# 16. right_wrist
# 17. left_pinky
# 18. right_pinky
# 19. left_index
# 20. right_index
# 21. left_thumb
# 22. right_thumb
# 23. left_hip
# 24. right_hip
# 25. left_knee
# 26. right_knee
# 27. left_ankle
# 28. right_ankle
# 29. left_heel
# 30. right_heel
# 31. left_foot_index
# 32. right_foot_index

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

class poseDetector():
    def __init__(self, mode=False, complexity=1, smooth=True, segmentation = False, smooth_segmentation = True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.complexity = complexity
        self.segmentation = segmentation
        self.smooth_segmentation = smooth_segmentation
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth, self.segmentation,
                                     self.smooth_segmentation, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        if img is None:
            return img

        img = ResizeWithAspectRatio(img, width=500)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if (self.results.pose_landmarks):
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        landmarkList = []
        for id, landmark in enumerate(self.results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            landmarkList.append([id, cx, cy])
            if draw:
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return landmarkList

def main():
    cap = cv2.VideoCapture('PoseVideos/6.mp4')
    previousTime = 0
    currentTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        if img is not None:
            img = detector.findPose(img)
            landmarkList = detector.findPosition(img)
            print(landmarkList[14])

            # FPS Measure
            currentTime = time.time()
            fps = 1 / (currentTime - previousTime)
            previousTime = currentTime
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

            cv2.imshow("Image", img)  # Display Image
            cv2.waitKey(1)  # Refresh
        else:
            break


if __name__ == "__main__":
    main()