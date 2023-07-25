<div align=center>
  <h1 align=center>SightVision</h1>
  <p align=center>Computer vision package that makes its easy to run Image processing and AI functions. At the core it uses OpenCV and Mediapipe libraries.</p>
</div>
<img src="git-images/sightvision.jpg" alt="Snake logo">

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
- [x] Python: Make sure you have Python installed on your system. SightVision requires Python to run, and it is recommended to use Python 3.x.
- [x] pip: Check if you have pip installed. Pip is the package manager for Python and is needed to install SightVision and its dependencies.
- [x] Operating System: SightVision is compatible with various operating systems, including Windows, macOS, and Linux. Ensure that you are using a supported operating system.
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
import sightvision
import cv2

cap = cv2.VideoCapture(0)
detector = sightvision.FaceDetector()

while True:
    success, frame = cap.read()
    frame, bboxs = detector.findFaces(frame)

    # Exit the application if the `q` key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

    cv2.imshow("Face detector", frame)
    cv2.waitKey(1)
```

<h2><a id="module">Available Modules 🧩</a></h2>

[Complete documentarion](https://github.com/rexionmars/SightVision/wiki)
- [x] Real-time Face Detection
- [X] FPS
- [x] FaceMash
- [x] Classification Module
- [x] PID Module
- [x] Pose Estimation
- [x] Serial Module
- [x] Face Mesh
- [x] Plot
- [x] Selfie Segmentations
- [x] Hand Tracking
- [x] Classification
- [x] Color Detection


## Sponsor the project

If you find this project useful and would like to support its ongoing development, consider becoming a sponsor. You can make a one-time or recurring donation and help keep this project alive.

[![Sponsor this project](https://img.shields.io/badge/GitHub%20Sponsors-Sponsor%20this%20project-red.svg)](https://github.com/sponsors/rexionmars)

Contact: [opensource.leonardi@gmail.com](mailto:opensource.leonardi@gmail.com)
