import logging
from enum import IntEnum
from time import time
from typing import cast

import numpy as np
from simple_pid import PID

from tello_aruco_nav.modules.tello import Tello
from tello_aruco_nav.schemas.map import MarkerData
from tello_aruco_nav.schemas.mission import MissionData

logger = logging.getLogger("drone_controller")

RC_UPDATE_TIME = 0.1
POSITION_TIMEOUT_DELAY = 1.0
HORIZONTAL_SPEED = 30
VERTICAL_SPEED = 20
MIN_VERTICAL_DISTANCE = 0.1

# X forward, Y right, Z down


class DroneState(IntEnum):
    IDLE = 0
    TAKING_OFF = 1
    FLYING = 2
    WAITING = 3
    LANDING = 4


class DroneController:
    def __init__(self, tello: Tello, mission: MissionData, markers: list[MarkerData]):
        self.__tello = tello
        self.__points = mission.waypoints
        self.__markers_map = {m.id: m for m in markers}
        self.__curr_wp_idx = -2
        self.__wait_start: float | None = None
        self.__rc_update_time = 0.0
        self.__last_wp_dist_v = 0.0
        self.__last_wp_delta_v = 0.0
        self.__dist_pid = PID(
            Kp=-60.0,
            Ki=6.0,
            Kd=12.0,
            setpoint=0.0,
            output_limits=(-60.0, 60.0),
        )
        self.__alt_pid = PID(
            Kp=50.0,
            Ki=2.0,
            Kd=0.0,
            setpoint=0.0,
            output_limits=(-60.0, 60.0),
        )

    def start(self):
        self.__curr_wp_idx = -1
        logger.info("Mission started")

    def stop(self):
        self.__curr_wp_idx = len(self.__points)
        logger.info("Mission stopping")

    @property
    def state(self):
        if self.__curr_wp_idx == -2:
            return DroneState.IDLE
        if self.__curr_wp_idx == -1:
            return DroneState.TAKING_OFF
        elif self.__curr_wp_idx == len(self.__points):
            return DroneState.LANDING
        elif self.__wait_start is not None:
            return DroneState.WAITING
        else:
            return DroneState.FLYING

    @property
    def waypoint_index(self):
        return self.__curr_wp_idx

    @property
    def current_marker(self):
        if self.__curr_wp_idx == -1 or self.__curr_wp_idx == len(self.__points):
            return None
        return self.__markers_map[self.__points[self.__curr_wp_idx].marker_id]

    @property
    def waypoint_distance(self):
        return self.__last_wp_dist_v

    @property
    def waypoint_altitude_delta(self):
        return self.__last_wp_delta_v

    @property
    def waypoint_wait_time(self):
        if self.__wait_start is None:
            return None

        return (
            self.__points[self.__curr_wp_idx].delay_after - time() + self.__wait_start
        )

    def update(self, position: np.ndarray | None, rotation_matrix: np.ndarray | None):
        if self.__curr_wp_idx == -2:
            return
        elif self.__curr_wp_idx == -1:
            if self.__tello.is_busy:
                return

            if not self.__tello.is_flying:
                logger.info("Taking off")
                self.__tello.takeoff()
            else:
                logger.info("Took off")
                self.__curr_wp_idx += 1
            return
        elif self.__curr_wp_idx == len(self.__points):
            if self.__tello.is_busy:
                return

            if self.__tello.is_flying:
                logger.info("Landing")
                self.__tello.send_rc_control(0, 0, 0, 0)
                self.__tello.land()
            else:
                logger.info("Landed")
                self.__curr_wp_idx = -2
            return

        if position is None or rotation_matrix is None:
            if time() - self.__rc_update_time > POSITION_TIMEOUT_DELAY:
                self.__tello.send_rc_control(0, 0, 0, 0)
                self.__rc_update_time = time()
            return

        wp = self.__points[self.__curr_wp_idx]
        marker = self.__markers_map[wp.marker_id]
        center = np.array(marker.center)
        center[1] = -wp.altitude

        dist_h = cast(float, np.linalg.norm(center[[0, 2]] - position[[0, 2]]))
        delta_v = self.__tello.height + center[1]
        self.__last_wp_dist_v = dist_h
        self.__last_wp_delta_v = delta_v

        speed_h = self.__dist_pid(dist_h)
        speed_h = speed_h if speed_h is not None else 0.0
        speed_v = self.__alt_pid(delta_v)
        speed_v = speed_v if speed_v is not None else 0.0
        logging.info(f"s={speed_h}, v={speed_v}")

        if self.__wait_start is None:
            if dist_h < wp.radius:
                self.__wait_start = time()
                logger.info(
                    f"Reached {self.__curr_wp_idx} marker. Waiting for {wp.delay_after}"
                )
        elif time() - self.__wait_start > wp.delay_after:
            self.__wait_start = None
            self.__curr_wp_idx += 1
            logger.info(f"Going to next point {self.__curr_wp_idx}")

        if time() - self.__rc_update_time > RC_UPDATE_TIME:
            self.__rc_update_time = time()
            delta = (center - position) @ rotation_matrix.T
            delta_xz = delta[[0, 2]]
            rc_control = delta_xz * (speed_h / np.linalg.norm(delta_xz))

            self.__tello.send_rc_control(
                int(rc_control[0]),
                int(rc_control[1]),
                int(speed_v),
                0,
            )
