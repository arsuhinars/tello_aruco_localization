import asyncio
import typing
from enum import IntEnum

import numpy as np
from imgui_bundle import imgui

from tello_aruco_nav.modules.mission_controller import MissionController
from tello_aruco_nav.modules.tello import Tello, TelloConnectionState
from tello_aruco_nav.modules.tello_controller import TelloController, TelloState
from tello_aruco_nav.schemas.map import MarkerData

if typing.TYPE_CHECKING:
    from tello_aruco_nav.modules.ui import Ui


class FlightMode(IntEnum):
    MANUAL = 0
    FOLLOW = 1
    MISSION = 2

    def __str__(self):
        match self:
            case self.MANUAL:
                return "Manual"
            case self.FOLLOW:
                return "Follow"
            case self.MISSION:
                return "Mission"


class FlightController:
    def __init__(
        self,
        markers: list[MarkerData],
        tello: Tello,
        mission_controller: MissionController,
        controller: TelloController,
    ):
        self.__markers_map = {m.id: m for m in markers}
        self.__tello = tello
        self.__controller = controller
        self.__mission_controller = mission_controller
        self.__mode = FlightMode.MANUAL
        self.__target_marker_id: int | None = None
        self.__target_altitude = 1.0

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, value: FlightMode):
        self.__mode = value

    @property
    def target_marker_id(self):
        return self.__target_marker_id

    @target_marker_id.setter
    def target_marker_id(self, value: int | None):
        self.__target_marker_id = value

        if value is None:
            self.__controller.target_pos = None
        else:
            x, _, z = self.__markers_map[value].center
            self.__controller.target_pos = np.array([x, self.__target_altitude, z])

    @property
    def target_altitude(self):
        return self.__target_altitude

    @target_altitude.setter
    def target_altitude(self, value: float):
        self.__target_altitude = value
        if self.__target_marker_id is not None:
            x, _, z = self.__markers_map[self.__target_marker_id].center
            self.__controller.target_pos = np.array([x, self.__target_altitude, z])

    def on_flight_button_clicked(self):
        match self.__mode:
            case FlightMode.MANUAL | FlightMode.FOLLOW:
                self.__controller.state = (
                    TelloState.TAKEOFF
                    if self.__controller.state
                    in [
                        TelloState.IDLE,
                        TelloState.TAKEOFF,
                    ]
                    else TelloState.LANDING
                )
            case FlightMode.MISSION:
                if self.__mission_controller.is_started:
                    self.__mission_controller.stop()
                else:
                    self.__mission_controller.start()

    async def run(self, ui: "Ui"):
        while True:
            pressed_keys, down_keys = await ui.on_keys_update()

            if self.__tello.connection_state != TelloConnectionState.CONNECTED:
                await asyncio.sleep(0.0)
                continue

            if imgui.Key.space in pressed_keys:
                self.on_flight_button_clicked()

            control = [0, 0, 0, 0]
            has_control = False
            if imgui.Key.w in down_keys:
                control[1] += 40
                has_control = True
            if imgui.Key.a in down_keys:
                control[0] -= 40
                has_control = True
            if imgui.Key.s in down_keys:
                control[1] -= 40
                has_control = True
            if imgui.Key.d in down_keys:
                control[0] += 40
                has_control = True
            if imgui.Key.left_shift in down_keys:
                control[2] += 40
                has_control = True
            if imgui.Key.left_ctrl in down_keys:
                control[2] -= 40
                has_control = True
            if imgui.Key.q in down_keys:
                control[3] -= 40
                has_control = True
            if imgui.Key.e in down_keys:
                control[3] += 40
                has_control = True
            match self.__mode:
                case FlightMode.MANUAL:
                    self.__controller.manual_control = tuple(control)
                case FlightMode.FOLLOW | FlightMode.MISSION:
                    self.__controller.manual_control = (
                        tuple(control) if has_control else None
                    )

            if imgui.Key.escape in pressed_keys:
                self.__tello.emergency()
                self.__mode = FlightMode.MANUAL
                self.__mission_controller.reset()
                self.__controller.state = TelloState.IDLE
                self.__controller.manual_control = None
                self.__controller.target_pos = None
                self.__target_altitude = 1.0
                self.__target_marker_id = None

            await asyncio.sleep(0.0)
