"""
Face Detection Module
Copyright (c) 2022 Leonardi Melo
"""
import cv2
import mediapipe as mp

from typing import Tuple, List, Dict, Any, Union, Optional

from ..utils.basics import rounded_rectangle


class FaceDetector:
    """
    Class for detecting faces in an image using the MediaPipe Face Detection model.
    """

    def __init__(self, min_detection_confidense=0.5):
        self.min_detection_confidense = min_detection_confidense
        self.media_pipe_face_Fetection = mp.solutions.face_detection
        self.media_pipe_draw = mp.solutions.drawing_utils
        self.face_detection = self.media_pipe_face_Fetection.FaceDetection(self.min_detection_confidense)

    def draw_detections(
            self,
            frame: object,
            bbox: Tuple[int, int, int, int],
            x: int,
            y: int,
            cx: int,
            cy: int,
            detection: object,
            view_mode: int,
            color=(25, 220, 255),
            thickness=1,
            external_info=False,
            internal_info=False,
            debug=False,
    ):
        """
        Draws the detections on the frame based on the specified parameters.

        Args:
            frame (numpy.ndarray): The input frame.
            bbox (tuple): The bounding box coordinates.
            x (int): The x-coordinate of the detection.
            y (int): The y-coordinate of the detection.
            cx (int): The x-coordinate of the center of the detection.
            cy (int): The y-coordinate of the center of the detection.
            detection (object): The detection object.
            view_mode (int): The display mode number.
            color (tuple, optional): The color of the drawn elements. Defaults to config.COLOR_LIGHT_YELLOW.
            thickness (int, optional): The thickness of the drawn elements. Defaults to 1.
        Raises:
            Exception: If the display mode number is not 1 or 2.
        Returns:
            None
        """

        padding_top = 17
        padding_left = 5
        bbox_confidence = detection.location_data.relative_bounding_box
        relative_keypoints = detection.location_data.relative_keypoints

        if view_mode not in [1, 2]:
            raise Exception("The display mode number reported does not exist, please select 1 or 2")

        match view_mode:
            case 1:
                rounded_rectangle(frame, bbox, 20, 1, 0, color, color)

            case 2:
                cv2.line(frame, (cx - 45, cy - 45), (x - 30, y - 30), color, thickness)
                cv2.circle(frame, (cx, cy), 64, color, thickness)

        # Draw acuracy in internal box
        if internal_info == True:
            cv2.putText(frame, f"{int(detection.score[0] * 100)}% Acuracy", (bbox[0] + padding_left, bbox[1] + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (155, 155, 155), 1)

        # Draw acuracy in external box
        if external_info == True:
            cv2.line(frame, (x, y), (x - 30, y - 30), color, thickness)
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

        if debug == True:
            print(f"Detection:\n{detection}")
            # Draw the bounding box coordinates
            cv2.putText(frame, f"{int(detection.score[0] * 100)}% Acuracy",
                        (bbox[0] + padding_left, bbox[1] + padding_top), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (155, 155, 155), 1)
            cv2.putText(frame, f"Xmin: {bbox_confidence.xmin:.4f}", (bbox[0] + padding_left, bbox[1] + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (155, 155, 155), 1)
            cv2.putText(frame, f"Ymin: {bbox_confidence.ymin:.4f}", (bbox[0] + padding_left, bbox[1] + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (155, 155, 155), 1)
            cv2.putText(frame, f"Width: {bbox_confidence.width:.4f}", (bbox[0] + padding_left, bbox[1] + 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (155, 155, 155), 1)
            cv2.putText(frame, f"Height: {bbox_confidence.height:.4f}", (bbox[0] + padding_left, bbox[1] + 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (155, 155, 155), 1)

            # Draw relative keypoints
            cv2.putText(frame, f"Left Eye: {relative_keypoints[0].x:.4f}, {relative_keypoints[0].y:.4f}",
                        (bbox[0] + padding_left, bbox[1] + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (155, 155, 155), 1)
            cv2.putText(frame, f"Right Eye: {relative_keypoints[1].x:.4f}, {relative_keypoints[1].y:.4f}",
                        (bbox[0] + padding_left, bbox[1] + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (155, 155, 155), 1)

            # Draw precision label based on aproach
            if detection.score[0] > 0.85:
                cv2.putText(frame, "High Precision", (bbox[0] + padding_left, bbox[1] + 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (155, 155, 155), 1)
            elif detection.score[0] > 0.70:
                cv2.putText(frame, "Medium Precision", (bbox[0] + padding_left, bbox[1] + 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (155, 155, 155), 1)
            else:
                cv2.putText(frame, "Low Precision", (bbox[0] + padding_left, bbox[1] + 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (155, 155, 155), 1)

    def find_faces(self,
                   frame,
                   view_mode=1,
                   draw=True,
                   color=(25, 220, 255),
                   thickness=1,
                   external_info=False,
                   internal_info=False,
                   debug=False):
        """
        Finds faces in the given frame using the face detection model.

        Args:
            frame (numpy.ndarray): The input frame in BGR format.
            view_mode (int, optional): The view mode for drawing the detections. Defaults to 1.
            draw (bool, optional): Whether to draw the detections on the frame. Defaults to True.
            color (tuple, optional): The color for drawing the detections. Defaults to (25, 220, 255).

        Returns:
            tuple: A tuple containing the modified frame with detections and a list of bounding boxes.
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(image_rgb)
        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bbox_confidence = detection.location_data.relative_bounding_box
                ih, iw, ic = frame.shape

                bbox = (
                    int(bbox_confidence.xmin * iw),
                    int(bbox_confidence.ymin * ih),
                    int(bbox_confidence.width * iw),
                    int(bbox_confidence.height * ih),
                )

                cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)
                x = bbox[0]
                y = bbox[1]

                bbox_info = {
                    "id": id,
                    "bbox": bbox,
                    "score": detection.score,
                    "center": (cx, cy),
                }
                bboxs.append(bbox_info)

                if draw:
                    self.draw_detections(frame, bbox, x, y, cx, cy, detection, view_mode, color, thickness,
                                         external_info, internal_info, debug)

        return frame, bboxs
