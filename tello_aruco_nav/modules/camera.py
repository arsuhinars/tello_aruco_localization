from abc import ABC, abstractmethod
from typing import cast

import cv2
import numpy as np

from tello_aruco_nav.common.exceptions import (
    CvCameraInitException,
    CvCameraReadFailedException,
)
from tello_aruco_nav.modules.tello import Tello


class BaseCamera(ABC):
    @abstractmethod
    def read_image(self) -> np.ndarray | None: ...

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
        self.__tello = tello

    def read_image(self):
        return self.__tello.read_next_frame()

    def release(self):
        self.__tello.stream_off()
