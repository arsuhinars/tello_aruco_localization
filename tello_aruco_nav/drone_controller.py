from time import time

import numpy as np
from djitellopy import Tello

from tello_aruco_nav.settings import ArucoCenter, MissionData
from tello_aruco_nav.utils import console

RC_UPDATE_TIME = 0.1
HORIZONTAL_SPEED = 30
VERTICAL_SPEED = 20
MIN_VERTICAL_DISTANCE = 0.1

# X forward, Y right, Z down


class DroneController:
    def __init__(self, tello: Tello, mission: MissionData, markers: list[ArucoCenter]):
        self.__tello = tello
        self.__points = mission.waypoints
        self.__markers_map = {m.id: m for m in markers}
        self.__curr_wp_idx = -1
        self.__wait_start: float | None = None
        self.__rc_update_time = 0.0

    def update(self, position: np.ndarray, rotation_matrix: np.ndarray):
        if self.__curr_wp_idx == -1:
            console.log("Takeoff")
            self.__tello.takeoff()
            self.__curr_wp_idx += 1
            return
        elif self.__curr_wp_idx == len(self.__points):
            console.log("Land")
            self.__tello.send_rc_control(0, 0, 0, 0)
            self.__tello.land()
            self.__curr_wp_idx += 1
            return
        elif self.__curr_wp_idx > len(self.__points):
            self.__tello.send_rc_control(0, 0, 0, 0)
            return

        if position is None or rotation_matrix is None:
            # TODO: magic number
            if time() - self.__rc_update_time > 1.0:
                self.__tello.send_rc_control(0, 0, 0, 0)
                self.__rc_update_time = time()
            return

        wp = self.__points[self.__curr_wp_idx]
        marker = self.__markers_map[wp.marker_id]
        center = np.array(marker.center)
        center[1] = -wp.altitude

        d = np.linalg.norm(position[[0, 2]] - center[[0, 2]])
        if self.__wait_start is None:
            if d < wp.radius:
                self.__wait_start = time()
                console.log(
                    f"Reached {self.__curr_wp_idx} marker. Waiting for {wp.delay_after}"
                )
        elif time() - self.__wait_start > wp.delay_after:
            self.__wait_start = None
            self.__curr_wp_idx += 1
            console.log(f"Going to next point {self.__curr_wp_idx}")

        if time() - self.__rc_update_time > RC_UPDATE_TIME:
            self.__rc_update_time = time()
            delta = (center - position) @ rotation_matrix.T
            delta_l = np.linalg.norm(delta)
            control = delta * (HORIZONTAL_SPEED / delta_l)

            self.__tello.send_rc_control(
                int(control[0]),
                int(control[2]),
                0
                if abs(delta[1]) < MIN_VERTICAL_DISTANCE
                else (-VERTICAL_SPEED if delta[1] > 0 else VERTICAL_SPEED),
                0,
            )
