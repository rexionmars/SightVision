import cv2

from sightvision.module.face_detection import FaceDetector

cap = cv2.VideoCapture(0)
detector = FaceDetector()

while True:
    success, frame = cap.read()
    frame, bboxs = detector.find_faces(frame, view_mode=1, external_info=True, debug=False)

    # Exit the application if the `q` key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

    cv2.imshow("Face detector", frame)
    cv2.waitKey(1)
