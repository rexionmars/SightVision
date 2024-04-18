from __future__ import annotations

import os

from distutils.core import setup
from setuptools import find_packages


def read(*names, **kwargs):
    """Read a file."""
    content = ""
    with open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


setup(
    name='sightvision',
    version='0.2.7',
    license='MIT',
    description='Computer vision package that makes its easy to run Image processing and AI functions.',
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author='Leonardi Melo',
    author_email='opensource.leonardi@gmail.com',
    url='https://github.com/rexionmars/SightVision',
    keywords=['ComputerVision', 'Tensorflow', 'MediaPipe', 'FaceDetection'],
    install_requires=['opencv-python', 'numpy', 'mediapipe'],
    python_requires='>=3.8',  # Requires any version >= 3.8
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',  # Specify which pyhton versions that you want to support
    ],
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    platforms="any",
)
