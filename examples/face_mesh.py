import cv2
from sightvision.module.face_mesh import FaceMeshDetector


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(max_faces=2)
    while True:
        success, img = cap.read()
        img, faces = detector.findface_mesh(img)
        if faces:
            print(faces[0])
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
