import cv2
import mediapipe as mp
import math

from sightvision.utils.basics import rounded_rectangle
from sightvision.configuration.constants import _RECTANGLE_DEFAULT_COLOR, _CIRCLE_DEFAULT_COLOR, _LINE_DEFAULT_SIZE


class PoseDetector:
    """
    Estimates Pose points of a human body using the mediapipe library.
    """

    def __init__(self, mode=False, smooth=True, detection_confidence=0.5, track_confidence=0.5):
        """
        Initializes the PoseDetector object.
        Args:
            mode: Inference mode of the Pose model.
            smooth: Smoothness of the landmarks.
            detectionCon: Minimum confidence required to detect a landmark.
            trackCon: Minimum confidence required to track a landmark.
        """

        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detection_confidence
        self.trackCon = track_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def find_pose(self, img, draw=True):
        """
        Finds the pose landmarks in the image.
        
        Args:
            img: Image to find the pose landmarks.
            draw: Flag to draw the landmarks on the image.
        Returns:
            Image with or without the landmarks."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)

        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def find_position(self,
                      img,
                      draw=True,
                      bboxWithHands=False,
                      circle_color=_CIRCLE_DEFAULT_COLOR,
                      circle_size=2,
                      rect_color=_RECTANGLE_DEFAULT_COLOR,
                      rect_size=_LINE_DEFAULT_SIZE):
        self.lmList = []
        self.bboxInfo = {}

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                self.lmList.append([id, cx, cy, cz])

            # Bounding Box
            ad = abs(self.lmList[12][1] - self.lmList[11][1]) // 2
            if bboxWithHands:
                x1 = self.lmList[16][1] - ad
                x2 = self.lmList[15][1] + ad
            else:
                x1 = self.lmList[12][1] - ad
                x2 = self.lmList[11][1] + ad

            y2 = self.lmList[29][2] + ad
            y1 = self.lmList[1][2] - ad
            bbox = (x1, y1, x2 - x1, y2 - y1)
            cx, cy = bbox[0] + (bbox[2] // 2), \
                     bbox[1] + bbox[3] // 2

            self.bboxInfo = {"bbox": bbox, "center": (cx, cy)}

            if draw:
                rounded_rectangle(
                    img,
                    bbox,
                    lenght_of_corner=20,
                    thickness_of_line=3,
                    radius_corner=1,
                    color_rectangle=rect_color,
                )
                cv2.circle(img, (cx, cy), circle_size, circle_color, cv2.FILLED)

        return self.lmList, self.bboxInfo

    def find_angle(self,
                   img,
                   p1,
                   p2,
                   p3,
                   draw=True,
                   circle_color=_CIRCLE_DEFAULT_COLOR,
                   circle_size=2,
                   line_color=_RECTANGLE_DEFAULT_COLOR,
                   line_size=_LINE_DEFAULT_SIZE):
        """
        Finds the angle between three points.
        
        Args:
            img: Image to draw the angle.
            p1: Point 1.
            p2: Point 2. The angle is calculated from this point.
            p3: Point 3.
            draw: Flag to draw the angle on the image.
        Returns:
            The angle between the three points."""

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), line_color, line_size)
            cv2.line(img, (x3, y3), (x2, y2), line_color, line_size)
            cv2.circle(img, (x1, y1), 10, circle_color, cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, circle_color, circle_size)
            cv2.circle(img, (x2, y2), 10, circle_color, cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, circle_color, circle_size)
            cv2.circle(img, (x3, y3), 10, circle_color, cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, circle_color, circle_size)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 0.5)
        return angle

    def find_distance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

    def angleCheck(self, myAngle, targetAngle, addOn=20):
        return targetAngle - addOn < myAngle < targetAngle + addOn
