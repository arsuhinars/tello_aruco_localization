import asyncio
import logging
import signal
from time import time

import numpy as np
import pygame as pg

from tello_aruco_nav.common.exceptions import (
    TelloDisconnectedException,
    TelloErrorException,
    TelloFailedConnectException,
)
from tello_aruco_nav.common.utils import console, euler_from_matrix, load_json
from tello_aruco_nav.modules.aruco_localization import ArucoLocalization
from tello_aruco_nav.modules.camera import TelloCamera
from tello_aruco_nav.modules.gui import (
    WINDOW_FRAMERATE,
    AlignHorizontal,
    AlignVertical,
    Gui,
)
from tello_aruco_nav.modules.tello import Tello, TelloConnectionState
from tello_aruco_nav.modules.tello_controller import TelloController, TelloState
from tello_aruco_nav.schemas.calibration import CalibrationData
from tello_aruco_nav.schemas.map import MapData
from tello_aruco_nav.schemas.mission import MissionData

logger = logging.getLogger("flight")


def run_flight(
    map_file: str = "map.json",
    mission_file: str | None = None,
    calibration_file: str = "calibration.json",
):
    FlightRunner(map_file, mission_file, calibration_file).run()


class FlightRunner:
    def __init__(self, map_file: str, mission_file: str | None, calibration_file: str):
        mission_data = (
            load_json(MissionData, mission_file)
            if mission_file is not None
            else MissionData(waypoints=[])
        )
        calibration_data = load_json(CalibrationData, calibration_file)
        map_data = load_json(MapData, map_file)

        self.__tello = Tello()
        self.__camera = TelloCamera(self.__tello)
        self.__localizer = ArucoLocalization(
            map_data.markers,
            calibration_data.get_np_matrix(),
            calibration_data.get_np_dist_coeffs(),
            calibration_data.rotation,
        )
        self.__controller = TelloController(
            self.__tello,
            map_data.markers,
            calibration_data.pid_x,
            calibration_data.pid_y,
            calibration_data.pid_z,
        )
        self.__gui = Gui()

        self.__is_running = False
        self.__camera_img: np.ndarray | None = None
        self.__camera_gray_img: np.ndarray | None = None
        self.__aruco_pos: np.ndarray | None = None
        self.__aruco_rot: np.ndarray | None = None

    def run(self):
        logger.info("Starting")

        self.__is_running = True

        def stop(signum, frame):
            self.stop()

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)

        loop = asyncio.new_event_loop()
        self.__task = loop.create_task(self.__main_loop())
        try:
            loop.run_until_complete(self.__task)
        except asyncio.CancelledError:
            ...
        except Exception:
            logger.exception("Exception occurred during main loop")

        self.__gui.stop()
        self.__camera.release()
        console.log("Stopping")
        if self.__tello.is_flying:
            self.__tello.emergency()
        self.__tello.disconnect()

        logger.info("Stopped")

    def stop(self):
        self.__is_running = False
        self.__task.cancel()

    async def __main_loop(self):
        asyncio.create_task(self.__gui_loop())

        while self.__is_running:
            try:
                await self.__tello.connect()
                await self.__tello.stream_on()

                try:
                    controller_task = asyncio.create_task(self.__controller.run())
                    await self.__flight_loop()
                except (TelloDisconnectedException, TelloErrorException):
                    logger.error("Reconnecting in 1 s.")
                finally:
                    self.__camera_img = None
                    self.__camera_gray_img = None
                    controller_task.cancel()
            except (
                TelloFailedConnectException,
                TelloDisconnectedException,
                TelloErrorException,
            ):
                logger.error("Retrying in 1 s.")

            await asyncio.sleep(1.0)

    async def __flight_loop(self):
        while (
            self.__is_running
            and self.__tello.connection_state != TelloConnectionState.DISCONNECTED
        ):
            self.__camera_img = self.__camera.read_image()
            self.__camera_gray_img, self.__aruco_pos, self.__aruco_rot = (
                self.__localizer.update(self.__camera_img)
            )
            self.__controller.feed_location(self.__aruco_pos, self.__aruco_rot)
            await asyncio.sleep(0.0)

    async def __gui_loop(self):
        self.__gui.run()

        last_frame_time = time()
        frame_delay = 1.0 / WINDOW_FRAMERATE
        while self.__is_running:
            if not self.__gui.is_running:
                self.stop()
                break

            if self.__tello.connection_state == TelloConnectionState.CONNECTED:
                if self.__gui.is_key_just_pressed(pg.K_SPACE):
                    if self.__controller.state not in [
                        TelloState.IDLE,
                        TelloState.LANDING,
                    ]:
                        self.__controller.state = TelloState.LANDING
                        self.__controller.manual_control = None
                        self.__controller.set_target_marker_id(None, None)
                    else:
                        self.__controller.state = TelloState.TAKEOFF

                control = [0, 0, 0, 0]
                has_control = False
                if self.__gui.is_key_down(pg.K_w):
                    control[1] += 40
                    has_control = True
                if self.__gui.is_key_down(pg.K_a):
                    control[0] -= 40
                    has_control = True
                if self.__gui.is_key_down(pg.K_s):
                    control[1] -= 40
                    has_control = True
                if self.__gui.is_key_down(pg.K_d):
                    control[0] += 40
                    has_control = True
                if self.__gui.is_key_down(pg.K_LSHIFT):
                    control[2] += 40
                    has_control = True
                if self.__gui.is_key_down(pg.K_LCTRL):
                    control[2] -= 40
                    has_control = True
                if self.__gui.is_key_down(pg.K_q):
                    control[3] -= 40
                    has_control = True
                if self.__gui.is_key_down(pg.K_e):
                    control[3] += 40
                    has_control = True
                self.__controller.manual_control = (
                    tuple(control) if has_control else None
                )

                if self.__gui.is_key_just_pressed(pg.K_ESCAPE):
                    self.__tello.emergency()
                    self.__controller.state = TelloState.IDLE
                    self.__controller.manual_control = None
                    self.__controller.set_target_marker_id(None, None)

            self.__gui.push_image(self.__camera_img)

            if self.__aruco_pos is not None:
                self.__gui.push_text(
                    f"x={self.__aruco_pos[0]:.2f}, y={self.__aruco_pos[1]:.2f}, z={self.__aruco_pos[2]:.2f}",
                    AlignHorizontal.LEFT,
                    AlignVertical.TOP,
                )
            if self.__aruco_rot is not None:
                euler = euler_from_matrix(self.__aruco_rot)
                self.__gui.push_text(
                    f"pitch={euler[0]:.0f}, yaw={euler[1]:.0f}, roll={euler[2]:.0f}",
                    AlignHorizontal.LEFT,
                    AlignVertical.TOP,
                )

            self.__gui.push_text(
                f"bat={self.__tello.battery:.0f}%",
                AlignHorizontal.RIGHT,
                AlignVertical.TOP,
            )
            self.__gui.push_text(
                f"h={self.__tello.height}", AlignHorizontal.RIGHT, AlignVertical.TOP
            )

            match self.__controller.state:
                case TelloState.IDLE:
                    self.__gui.push_text(
                        "idle", AlignHorizontal.CENTER, AlignVertical.TOP
                    )
                case TelloState.TAKEOFF:
                    self.__gui.push_text(
                        "taking off", AlignHorizontal.CENTER, AlignVertical.TOP
                    )
                case TelloState.GO_TO_MARKER:
                    marker_id = self.__controller.marker_id
                    marker_dist = self.__controller.marker_dist
                    marker_alt_delta = self.__controller.marker_alt_delta
                    self.__gui.push_text(
                        f"flying to marker with id={marker_id}",
                        AlignHorizontal.CENTER,
                        AlignVertical.TOP,
                    )
                    self.__gui.push_text(
                        f"dist={marker_dist:.2f}, delta_alt={marker_alt_delta:.2f}",
                        AlignHorizontal.CENTER,
                        AlignVertical.TOP,
                    )
                case TelloState.MANUAL_CONTROL:
                    control = self.__controller.manual_control
                    self.__gui.push_text(
                        "manual", AlignHorizontal.CENTER, AlignVertical.TOP
                    )
                    if control is not None:
                        self.__gui.push_text(
                            f"x={control[0]}, y={control[2]}, z={control[1]}",
                            AlignHorizontal.CENTER,
                            AlignVertical.TOP,
                        )
                case TelloState.WAITING:
                    self.__gui.push_text(
                        "waiting", AlignHorizontal.CENTER, AlignVertical.TOP
                    )
                case TelloState.LANDING:
                    self.__gui.push_text(
                        "landing", AlignHorizontal.CENTER, AlignVertical.TOP
                    )

            self.__gui.update()
            await asyncio.sleep(max(frame_delay - time() + last_frame_time, 0.0))
