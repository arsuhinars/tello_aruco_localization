import asyncio
import logging
from dataclasses import dataclass
from enum import IntEnum
from math import copysign
from typing import cast

import numpy as np
from simple_pid import PID

from tello_aruco_nav.common.utils import Float3
from tello_aruco_nav.modules.tello import Tello

logger = logging.getLogger("controller")

UPDATE_DELAY = 0.1
FLY_BY_OFFSET_SPEED = 80

# X forward, Y right, Z down


class TelloState(IntEnum):
    IDLE = 0
    TAKEOFF = 1
    GO_TO_POS = 2
    MANUAL_CONTROL = 3
    FLY_BY_OFFSET = 4
    WAITING = 5
    LANDING = 6


@dataclass
class PidState:
    current: float = 0.0
    target: float = 0.0
    control: float = 0.0


class TelloController:
    def __init__(
        self,
        tello: Tello,
        pid_x: Float3,
        pid_y: Float3,
        pid_z: Float3,
        rate_expo: Float3,
    ):
        self.__tello = tello
        self.__position: np.ndarray | None = None
        self.__rotation: np.ndarray | None = None
        self.__target_pos: np.ndarray | None = None
        self.__manual_controls: tuple[int, int, int, int] | None = None
        self.__last_stabilize_controls: tuple[int, int] | None = None
        self.__fly_by_offset: Float3 | None = None
        self.__marker_dist: float | None = None
        self.__marker_delta_alt = 0.0
        self.__state = TelloState.IDLE
        self.__takeoff_event = asyncio.Event()
        self.__land_event = asyncio.Event()
        self.__land_event.set()
        self.__flew_by_event = asyncio.Event()

        self.__pid_x = PID(*pid_x, output_limits=(-40.0, 40.0))
        self.__pid_x_state = PidState()
        self.__pid_y = PID(*pid_y, output_limits=(-60.0, 60.0), setpoint=1.0)
        self.__pid_y_state = PidState()
        self.__pid_z = PID(*pid_z, output_limits=(-40.0, 40.0))
        self.__pid_z_state = PidState()

        self.__rate_expo = rate_expo

    @property
    def state(self):
        return self.__state

    @state.setter
    def state(self, value: TelloState):
        self.__state = value

    @property
    def is_flying(self):
        return self.__state in [
            TelloState.GO_TO_POS,
            TelloState.WAITING,
            TelloState.MANUAL_CONTROL,
        ]

    async def on_take_off(self):
        await self.__takeoff_event.wait()

    async def on_landed(self):
        await self.__land_event.wait()

    async def on_flown(self):
        await self.__flew_by_event.wait()

    @property
    def marker_dist(self):
        return self.__marker_dist

    @property
    def marker_alt_delta(self):
        return self.__marker_delta_alt

    @property
    def manual_control(self):
        return self.__manual_controls

    @manual_control.setter
    def manual_control(self, control: tuple[int, int, int, int] | None):
        self.__manual_controls = control

    @property
    def target_pos(self):
        return self.__target_pos

    @target_pos.setter
    def target_pos(self, value: np.ndarray | None):
        self.__target_pos = value
        if value is not None:
            self.__pid_x.setpoint = value[0]
            self.__pid_y.setpoint = value[1]
            self.__pid_z.setpoint = value[2]

    @property
    def fly_by_offset(self):
        return self.__fly_by_offset

    @fly_by_offset.setter
    def fly_by_offset(self, value: Float3 | None):
        self.__fly_by_offset = value

    def feed_location(self, position: np.ndarray | None, rotation: np.ndarray | None):
        self.__position = position
        self.__rotation = rotation

    @property
    def pid_x(self):
        return self.__pid_x.tunings

    @pid_x.setter
    def pid_x(self, tunings: Float3):
        self.__pid_x.tunings = tunings

    @property
    def pid_x_state(self):
        return self.__pid_x_state

    @property
    def pid_y(self):
        return self.__pid_y.tunings

    @pid_y.setter
    def pid_y(self, tunings: Float3):
        self.__pid_y.tunings = tunings

    @property
    def pid_y_state(self):
        return self.__pid_y_state

    @property
    def pid_z(self):
        return self.__pid_z.tunings

    @pid_z.setter
    def pid_z(self, tunings: Float3):
        self.__pid_z.tunings = tunings

    @property
    def pid_z_state(self):
        return self.__pid_z_state

    @property
    def rate_expo(self):
        return self.__rate_expo

    @rate_expo.setter
    def rate_expo(self, value: Float3):
        self.__rate_expo = value

    async def run(self):
        while True:
            match self.__state:
                case TelloState.IDLE:
                    ...
                case TelloState.TAKEOFF:
                    logger.info("Taking off")
                    if not self.__tello.is_flying:
                        await self.__tello.takeoff()
                    self.__state = TelloState.WAITING
                    self.__takeoff_event.set()
                    self.__land_event.clear()
                    logger.info("Took off")
                case TelloState.LANDING:
                    logger.info("Landing")
                    if self.__tello.is_flying:
                        await self.__tello.land()
                    self.__land_event.set()
                    self.__takeoff_event.clear()
                    logger.info("Landed")
                    self.__state = TelloState.IDLE
                case TelloState.FLY_BY_OFFSET:
                    if self.__fly_by_offset is not None:
                        x, y, z = self.__fly_by_offset
                        logger.info(f"Fly by {x} {y} {z}")
                        await self.__tello.go(
                            int(x), int(y), int(z), FLY_BY_OFFSET_SPEED
                        )
                        logger.info("Done")
                    self.__flew_by_event.set()
                    await asyncio.sleep(0.0)
                    self.__flew_by_event.clear()
                    self.__state = TelloState.WAITING
                case _:
                    if self.__manual_controls is not None:
                        self.__state = TelloState.MANUAL_CONTROL
                    elif self.__target_pos is not None:
                        self.__state = TelloState.GO_TO_POS
                    else:
                        self.__state = TelloState.WAITING

            match self.__state:
                case TelloState.GO_TO_POS:
                    self.__stabilize_position()
                case TelloState.MANUAL_CONTROL:
                    if self.__manual_controls is not None:
                        self.__tello.send_rc_control(*self.__manual_controls)
                case TelloState.WAITING:
                    self.__tello.send_rc_control(0, 0, 0, 0)

            await asyncio.sleep(UPDATE_DELAY)

    def __stabilize_position(self):
        assert self.__target_pos is not None

        # if self.__position is None:
        curr_alt = self.__tello.height
        # else:
        # curr_alt = -self.__position[1]
        self.__marker_delta_alt = curr_alt - self.__target_pos[1]

        rc_up_down = self.__pid_y(curr_alt)
        rc_up_down = (
            0
            if rc_up_down is None
            else int(self.calc_rate_value(rc_up_down, self.__rate_expo[1]))
        )
        self.__pid_y_state.current = curr_alt
        self.__pid_y_state.target = self.__target_pos[1]
        self.__pid_y_state.control = rc_up_down

        if self.__position is not None and self.__rotation is not None:
            target_pos = self.__target_pos[[0, 2]]
            current_pos = self.__position[[0, 2]]
            self.__marker_dist = cast(float, np.linalg.norm(target_pos - current_pos))

            vec = np.array(
                [
                    self.__pid_x(self.__position[0]),
                    0.0,
                    self.__pid_z(self.__position[2]),
                ]
            )

            self.__pid_x_state.current = self.__position[0]
            self.__pid_x_state.target = target_pos[0]
            self.__pid_x_state.control = vec[0]

            self.__pid_z_state.current = self.__position[2]
            self.__pid_z_state.target = target_pos[1]
            self.__pid_z_state.control = vec[2]

            # vec @= self.__rotation

            rc_left_right = int(self.calc_rate_value(vec[0], self.__rate_expo[0]))
            rc_forward_backward = int(self.calc_rate_value(vec[2], self.__rate_expo[2]))

            self.__last_stabilize_controls = (rc_left_right, rc_forward_backward)
        # elif self.__last_stabilize_controls is not None:
        # rc_left_right = -self.__last_stabilize_controls[0]
        # rc_forward_backward = -self.__last_stabilize_controls[1]
        else:
            rc_left_right = 0
            rc_forward_backward = 0
            self.__marker_dist = None

        self.__tello.send_rc_control(
            rc_left_right,
            rc_forward_backward,
            rc_up_down,
            0,
        )

    @staticmethod
    def calc_rate_value(x: float, expo: float):
        return expo * 100.0 * ((x / 100.0) ** 2) * copysign(1.0, x) + x * (1.0 - expo)
