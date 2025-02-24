from abc import ABC, abstractmethod
from typing import cast

import cv2
import numpy as np
from djitellopy import Tello

from tello_aruco_nav.exceptions import (
    CvCameraInitException,
    CvCameraReadFailedException,
)


class BaseCamera(ABC):
    @abstractmethod
    def read_image(self) -> np.ndarray: ...

    @abstractmethod
    def release(self): ...


class CvCamera(BaseCamera):
    def __init__(self, camera_index: int):
        self.__capture = cv2.VideoCapture(camera_index)

        if not self.__capture.isOpened():
            raise CvCameraInitException()

    def read_image(self):
        result, img = self.__capture.read()

        if not result:
            raise CvCameraReadFailedException()

        return cast(np.ndarray, img)

    def release(self):
        self.__capture.release()


class TelloCamera(BaseCamera):
    def __init__(self, tello: Tello):
        self.__frame_reader = tello.get_frame_read()

    def read_image(self):
        return cast(np.ndarray, self.__frame_reader.frame)

    def release(self):
        self.__frame_reader.stop()
