import cv2
import mediapipe as mp
import time

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

cap = cv2.VideoCapture("PoseVideos/7.mp4")
previousTime = 0
currentTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2, color=(0, 255, 0))

while True:
    success, img = cap.read()
    img = ResizeWithAspectRatio(img, width=800)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLandmarks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLandmarks, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)
            for id, landmark in enumerate(faceLandmarks.landmark):
                ih, iw, ic = img.shape
                x, y = int(landmark.x * iw), int(landmark.y * ih)

    # FPS Measure
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

    cv2.imshow("Image", img)  # Display Image
    cv2.waitKey(1)  # Refresh