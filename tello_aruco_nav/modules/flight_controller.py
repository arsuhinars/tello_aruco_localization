from enum import IntEnum

from imgui_bundle import imgui

from tello_aruco_nav.modules.mission_controller import MissionController
from tello_aruco_nav.modules.tello import Tello, TelloConnectionState
from tello_aruco_nav.modules.tello_controller import TelloController, TelloState


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
        tello: Tello,
        mission_controller: MissionController,
        controller: TelloController,
    ):
        self.__tello = tello
        self.__controller = controller
        self.__mission_controller = mission_controller
        self.__mode = FlightMode.MANUAL

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, value: FlightMode):
        self.__mode = value

    @property
    def target_marker_id(self):
        return self.__controller.target_marker_id

    @target_marker_id.setter
    def target_marker_id(self, value: int | None):
        self.__controller.target_marker_id = value

    @property
    def target_altitude(self):
        return self.__controller.target_altitude

    @target_altitude.setter
    def target_altitude(self, value: float):
        self.__controller.target_altitude = value

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

    def trigger_imgui_update(self):
        if self.__tello.connection_state != TelloConnectionState.CONNECTED:
            return

        if imgui.is_key_pressed(imgui.Key.space, False):
            self.on_flight_button_clicked()

        control = [0, 0, 0, 0]
        has_control = False
        if imgui.is_key_down(imgui.Key.w):
            control[1] += 40
            has_control = True
        if imgui.is_key_down(imgui.Key.a):
            control[0] -= 40
            has_control = True
        if imgui.is_key_down(imgui.Key.s):
            control[1] -= 40
            has_control = True
        if imgui.is_key_down(imgui.Key.d):
            control[0] += 40
            has_control = True
        if imgui.is_key_down(imgui.Key.left_shift):
            control[2] += 40
            has_control = True
        if imgui.is_key_down(imgui.Key.left_ctrl):
            control[2] -= 40
            has_control = True
        if imgui.is_key_down(imgui.Key.q):
            control[3] -= 40
            has_control = True
        if imgui.is_key_down(imgui.Key.e):
            control[3] += 40
            has_control = True
        match self.__mode:
            case FlightMode.MANUAL:
                self.__controller.manual_control = tuple(control)
            case FlightMode.FOLLOW | FlightMode.MISSION:
                self.__controller.manual_control = (
                    tuple(control) if has_control else None
                )

        if imgui.is_key_pressed(imgui.Key.escape, False):
            self.__tello.emergency()
            self.__mode = FlightMode.MANUAL
            self.__mission_controller.reset()
            self.__controller.state = TelloState.IDLE
            self.__controller.manual_control = None
            self.__controller.target_marker_id = None
