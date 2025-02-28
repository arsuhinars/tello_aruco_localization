from enum import IntEnum
from time import time

import numpy as np
from djitellopy import Tello

from tello_aruco_nav.utils import rotation_matrix_euler


class LocalizationStatus(IntEnum):
    UNKNOWN = 0
    ONLINE = 1
    OFFLINE = 2


class TelloLocalization:
    def __init__(self, tello: Tello):
        self.__tello = tello
        self.__status = LocalizationStatus.UNKNOWN
        self.__delta_pos = np.array([0.0, 0.0, 0.0], np.float32)
        self.__last_pos: np.ndarray | None = None
        self.__real_pos: np.ndarray | None = None
        self.__home_pos: np.ndarray | None = None
        self.__last_update_time = time()

    def update(self, pos: np.ndarray | None, rot: np.ndarray | None):
        time_delta = time() - self.__last_update_time
        self.__last_update_time = time()

        if pos is None or rot is None:
            self.__status = (
                LocalizationStatus.OFFLINE
                if self.__status == LocalizationStatus.ONLINE
                else LocalizationStatus.UNKNOWN
            )

            rot_mtx = rotation_matrix_euler(
                self.__tello.get_pitch(),
                self.__tello.get_yaw(),
                self.__tello.get_roll(),
            )[:3, :3]

            speed = (
                np.array(
                    [
                        self.__tello.get_speed_y() / 100.0,
                        self.__tello.get_speed_z() / 100.0,
                        self.__tello.get_speed_x() / 100.0,
                    ],
                    np.float32,
                )
                @ rot_mtx
            )

            self.__delta_pos += speed * time_delta

            if self.__last_pos is not None:
                self.__real_pos = self.__last_pos + self.__delta_pos

            return

        if self.__status == LocalizationStatus.UNKNOWN:
            self.__home_pos = pos - self.__delta_pos
            self.__delta_pos.fill(0.0)
        elif self.__status == LocalizationStatus.OFFLINE:
            self.__delta_pos.fill(0.0)

        self.__status = LocalizationStatus.ONLINE
        self.__real_pos = pos

    @property
    def status(self):
        return self.__status

    @property
    def delta_pos(self):
        return self.__delta_pos

    @property
    def last_pos(self):
        return self.__last_pos

    @property
    def real_pos(self):
        return self.__real_pos

    @property
    def home_pos(self):
        return self.__home_pos
