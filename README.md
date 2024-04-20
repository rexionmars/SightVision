<div align=center>
  <h1 align=center>SightVision</h1>
  <p align=center>Computer vision package that makes its easy to run Image processing and AI functions. At the core it uses OpenCV and Mediapipe libraries.</p>
</div>
<img src=".github/img/sightvision.jpg" alt="Snake logo">

SightVision is a powerful Computer Vision library that provides various methods for face detection, hand detection, and other related functionalities. With SightVision, you can effortlessly integrate image analysis capabilities into your projects and applications.
## Get Started
1. [Install Guide](#install-guide)
2. [Using the SightVision Library](#use)
3. [Available Modules](#module)
4. [Contributing](CONTRIBUTING.md)
5. [License](LICENSE.md)

## Installation of the SightVision library via pip
The SightVision library is a powerful image processing and computer vision tool that enables object detection and analysis, face detection, and other features in images and videos. To make use of all the capabilities of SightVision in your project, follow the steps below to install it via pip.

## Requirements
Before starting the installation, please ensure that you meet the following requirements:
- [x] Hardware: Depending on the specific tasks and models you plan to use with SightVision, you might need sufficient computational resources, such as CPU and GPU capabilities.

## Step 1: Set up a Virtual Environment (optional)

While not strictly necessary, it is a good practice to create a virtual environment before installing new libraries on your system. This will help prevent dependency conflicts with other projects.

## Step 2: Installation via pip
Now that you have Python set up, let's install the SightVision library via pip. Open a terminal or command prompt and execute the following command:
```sh
pip install sightvision
```
Pip will start downloading the necessary files and installing the library. Please wait until the installation is successfully completed.

## Step 3: Verify the Installation

To ensure that the installation was successful, you can check if the SightVision library is accessible in your Python environment. Simply open a Python interpreter or a Jupyter Notebook and type the following:
```python
import sightvision

# If there are no import errors, the library is installed successfully.
print("SightVision library is accessible.")
```
Running the above code will import the SightVision library, and if there are no import errors, it confirms that the installation was successful. You are now ready to utilize the SightVision library in your Python projects and take advantage of its image processing and computer vision functionalities. Happy coding!

<h2><a id="use">Step 4: Using the SightVision Library</a></h2>


Now that the library is installed, you can start exploring its functionalities in your image processing and computer vision projects. Make sure to read the official SightVision documentation for detailed information on how to use each feature it offers.
<br>**Checkout more examples**: [Complete documentarion](https://github.com/rexionmars/SightVision/wiki)
#### Face Detection module
```python
import cv2

from sightvision.module.face_detection import FaceDetector

cap = cv2.VideoCapture(0)
detector = FaceDetector()

while True:
    success, frame = cap.read()
    frame, bboxs = detector.find_faces(frame, view_mode=1, external_info=True, debug=False)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

    cv2.imshow("Face detector", frame)
    cv2.waitKey(1)
```


#### Hand Tracking Module
```python
import cv2
from sightvision.module.hand_tracking import HandDetector


cap = cv2.VideoCapture(0)
detector = HandDetector(detection_confidence=0.8, max_hands=2)

while True:
    success, img = cap.read()
    hands, img = detector.find_hands(img)

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        bbox1 = hand1["bbox"]
        centerPoint1 = hand1['center']
        handType1 = hand1["type"]

        fingers1 = detector.fingersUp(hand1)

        if len(hands) == 2:
            # Hand 2
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            bbox2 = hand2["bbox"]
            centerPoint2 = hand2['center']
            handType2 = hand2["type"]

            fingers2 = detector.fingersUp(hand2)

            # Find distance
            length, info, img = detector.find_distance(lmList1[8][0:2], lmList2[8][0:2], img)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
```

<h2><a id="module">Available Modules</a></h2>

- [x] Real-time Face Detection
- [x] Hand Tracking
- [x] Pose Estimation
- [x] Face Mesh

<h2><a id="module">Available Utils Functions</a></h2>

- [X] Rounded Rectangle
- [x] Overlay PNG
- [x] Stack Images

## Sponsor the project

If you find this project useful and would like to support its ongoing development, consider becoming a sponsor. You can make a one-time or recurring donation and help keep this project alive.

[![Sponsor this project](https://img.shields.io/badge/GitHub%20Sponsors-Sponsor%20this%20project-red.svg)](https://github.com/sponsors/rexionmars)

Contact: [opensource.leonardi@gmail.com](mailto:opensource.leonardi@gmail.com)
