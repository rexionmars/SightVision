import time
import cv2


class FPS:
    """
    Helps in finding Frames Per Second and display on an OpenCV Image
    """

    def __init__(self):
        self.pTime = time.time()

    def update(self, frame=None, pos=(20, 50), color=(22, 75, 203), scale=2, thickness=1):
        """
        Update the frame rate
        :param frame: Frame to display on, can be left blank if only fps value required
        :param pos: Position on the FPS on the image
        :param color: Color of the FPS Value displayed
        :param scale: Scale of the FPS Value displayed
        :param thickness: Thickness of the FPS Value displayed
        :return:
        """
        cTime = time.time()
        try:
            fps = 1 / (cTime - self.pTime)
            self.pTime = cTime
            if frame is None:
                return fps
            else:
                cv2.putText(frame, f'FPS: {int(fps)}', pos, cv2.FONT_HERSHEY_PLAIN,
                            scale, color, thickness)
                return fps, frame
        except:
            return 0


def main():
    """
    Without Webcam
    """
    fpsReader = FPS()
    while True:
        time.sleep(0.025)  # add delay to get 40 Frames per second
        fps = fpsReader.update()
        print(fps)


def mainWebcam():
    """
    With Webcam
    """
    fpsReader = FPS()
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        fps, frame = fpsReader.update(frame)
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()