import cv2
import mediapipe as mp
import math

from sightvision.utils.basics import rounded_rectangle
from sightvision.configuration.constants import _RECTANGLE_DEFAULT_COLOR, _LINE_DEFAULT_SIZE


class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, min_track_confidence=0.5):
        """
        Args:
            mode: In static mode, detection is done on each image: slower
            maxHands: Maximum number of hands to detect
            detectionCon: Minimum Detection Confidence
            trackCon: Minimum Tracking Confidence
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.min_track_confidence = min_track_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=self.mode,
                                         max_num_hands=self.max_hands,
                                         min_detection_confidence=self.detection_confidence,
                                         min_tracking_confidence=self.min_track_confidence)
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lm_list = []

    def find_hands(self,
                   img,
                   draw=True,
                   flip_type=True,
                   color=_RECTANGLE_DEFAULT_COLOR,
                   line_size=_LINE_DEFAULT_SIZE):
        """
        Finds hands in a BGR image.
        
        Args:
            img: Image to find the hands in.
            draw: Flag to draw the output on the image.
            flipType: Flip the hand type.
        Returns:
            Image with or without drawings
            List of hands with landmarks"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        all_hands = []
        h, w, c = img.shape

        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                my_hand = {}
                my_land_mark_list = []
                x_list = []
                y_list = []
                id = handType.classification[0].index

                for _, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    my_land_mark_list.append([px, py, pz])
                    x_list.append(px)
                    y_list.append(py)

                # Bounding box
                xmin, xmax = min(x_list), max(x_list)
                ymin, ymax = min(y_list), max(y_list)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                my_hand["lmList"] = my_land_mark_list
                my_hand["bbox"] = bbox
                my_hand["center"] = (cx, cy)

                if flip_type:
                    if handType.classification[0].label == "Right":
                        my_hand["type"] = "Left"
                    else:
                        my_hand["type"] = "Right"
                else:
                    my_hand["type"] = handType.classification[0].label

                all_hands.append(my_hand)

                if draw:
                    self.mp_draw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)

                    rounded_rectangle(
                        img,
                        bbox,
                        lenght_of_corner=20,
                        thickness_of_line=1,
                        radius_corner=0,
                        color_rectangle=color,
                    )

                    cv2.putText(img, my_hand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                color, 1)
        if draw:
            return all_hands, img
        else:
            return all_hands

    def fingersUp(self, myHand):
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :return: List of which fingers are up
        """
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:
            fingers = []
            # Thumb
            if myHandType == "Right":
                if myLmList[self.tip_ids[0]][0] > myLmList[self.tip_ids[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tip_ids[0]][0] < myLmList[self.tip_ids[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if myLmList[self.tip_ids[id]][1] < myLmList[self.tip_ids[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def find_distance(self,
                      p1,
                      p2,
                      img=None,
                      circle_color=(25, 0, 255),
                      circle_size=3,
                      line_color=(25, 0, 255),
                      line_size=1):
        """
        Find the distance between two landmarks based on their 2D coordinates.

        Args:
            p1: Point1 (x1, y1)
            p2: Point2 (x2, y2)
            img: Image to draw on.
        Returns:
            Length of the line and the image with the line plotted
        """
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)

        if img is not None:
            cv2.circle(img, (x1, y1), circle_size, circle_color, cv2.FILLED)
            cv2.circle(img, (x2, y2), circle_size, circle_color, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), line_color, line_size)
            cv2.circle(img, (cx, cy), circle_size, circle_color, cv2.FILLED)
            return length, info, img
        else:
            return length, info
