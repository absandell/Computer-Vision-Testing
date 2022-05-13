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

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, refLandmarks=False, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refLandmarks = refLandmarks
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.refLandmarks, self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2, color=(0, 255, 0))

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.imgRGB is not None:
            self.results = self.faceMesh.process(self.imgRGB)
            faces = []
            if self.results.multi_face_landmarks:
                for faceLandmarks in self.results.multi_face_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img, faceLandmarks, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)

                    face = []
                    for id, landmark in enumerate(faceLandmarks.landmark):
                        ih, iw, ic = img.shape
                        x, y = int(landmark.x * iw), int(landmark.y * ih)
                        face.append([x,y])
                    faces.append(face)
        return img, faces

def main():
    cap = cv2.VideoCapture("PoseVideos/2.mp4")
    previousTime = 0
    currentTime = 0
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        if img is not None:
            img = ResizeWithAspectRatio(img, width=800)
            img, faces = detector.findFaceMesh(img)
            if len(faces) != 0:
                print(len(faces))
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