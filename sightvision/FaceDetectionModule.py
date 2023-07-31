import cv2
import mediapipe as mp

import config
from Utils import stackImages


class FaceDetector:
    """
    Find faces in realtime using the light weight model provided in the
    mediapipe library.
    """

    def __init__(self, minDetectionCon=0.5):
        """
        :param minDetectionCon: Minimum Detection Confidence Threshold
        """

        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def draw_detections(
        self,
        frame,
        bbox,
        x,
        y,
        cx,
        cy,
        detection,
        view_mode,
        color=config.COLOR_LIGHT_YELLOW,
        thickness=1,
    ):
        """
        Foo bar buzz
        """
        if view_mode not in [1, 2]:
            raise Exception(
                "The display mode number reported does not exist, please\
                select 1 or 2"
            )

        match view_mode:
            case 1:
                # Draw a rectangle
                cv2.line(frame, (x, y), (x - 30, y - 30), color, thickness)
                cv2.rectangle(frame, bbox, color, thickness)

            case 2:
                # Draw a circle
                cv2.line(frame, (cx - 45, cy - 45), (x - 30, y - 30), color, thickness)
                cv2.circle(frame, (cx, cy), 64, color, thickness)

        # Label for text acuracy
        cv2.rectangle(frame, (x - 145, y - 50, 115, 20), color, -1)
        cv2.putText(
            frame,
            f"{int(detection.score[0] * 100)}% Acuracy",
            (bbox[0] - 140, bbox[1] - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    def findFaces(self, frame, view_mode=1, draw=True, color=(25, 220, 255)):
        """
        Find faces in an image and return the bbox info

        :param frame: Image to find the faces in.
        :param view_mode: Mode of visualization (1 for rectangle, 2 for circle)
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings. Bounding Box list.
        """

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = frame.shape

                bbox = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )

                cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)
                x = bbox[0]
                y = bbox[1]

                bboxInfo = {
                    "id": id,
                    "bbox": bbox,
                    "score": detection.score,
                    "center": (cx, cy),
                }
                bboxs.append(bboxInfo)

                if draw:
                    self.draw_detections(
                        frame, bbox, x, y, cx, cy, detection, view_mode
                    )

        return frame, bboxs
