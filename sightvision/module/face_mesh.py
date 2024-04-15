import cv2
import mediapipe as mp
import math


class FaceMeshDetector:
    """
    Face Mesh Detector to find 468 Landmarks using the mediapipe library.
    Helps acquire the landmark points in pixel format
    """

    def __init__(self,
                 static_mode=False,
                 max_faces=2,
                 min_detection_confidence=0.5,
                 min_track_confidence=0.5,
                 color=(0, 255, 0)):
        """
        Initializes the Face Mesh Detector.
        Args:
            staticMode: In static mode, detection is done on each image: slower
            maxFaces: Maximum number of faces to detect
            min_detection_confidence: Minimum Detection Confidence
            min_track_confidence: Minimum Tracking Confidence
        """
        self.staticMode = static_mode
        self.max_faces = max_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_track_confidence = min_track_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=self.staticMode,
                                                    max_num_faces=self.max_faces,
                                                    min_detection_confidence=self.min_detection_confidence,
                                                    min_tracking_confidence=self.min_track_confidence)
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=0, color=color)

    def findface_mesh(self, img, draw=True):
        """
        Find the face landmarks in an Image of BGR color space.
        Args:
            img: Image to find the face landmarks in.
            draw: Flag to draw the output on the image.
        Returns:
            Image with or without drawings
            Landmark points in pixel format
        """
        self.img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(self.img_rgb)
        faces = []

        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                                                self.draw_spec, self.draw_spec)

                face = []
                for id, lm in enumerate(face_landmarks.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
        return img, faces

    def find_distance(self, p1, p2, img=None):
        """
        Find the distance between two landmarks based on their
        index numbers.
        :param p1: Point1
        :param p2: Point2
        :param img: Image to draw on.
        :param draw: Flag to draw the output on the image.
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        color = (22, 75, 203)

        if img is not None:
            cv2.circle(img, (x1, y1), 15, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, color, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), color, 3)
            cv2.circle(img, (cx, cy), 15, color, cv2.FILLED)
            return length, info, img
        else:
            return length, info
