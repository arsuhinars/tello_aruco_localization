import asyncio
import logging
import signal
from time import time

import numpy as np

from tello_aruco_nav.common.exceptions import (
    TelloDisconnectedException,
    TelloErrorException,
    TelloFailedConnectException,
)
from tello_aruco_nav.common.utils import console, euler_from_matrix, load_json
from tello_aruco_nav.modules.aruco_localization import ArucoLocalization
from tello_aruco_nav.modules.camera import TelloCamera
from tello_aruco_nav.modules.flight_controller import FlightController, FlightMode
from tello_aruco_nav.modules.hud import (
    HUD_FRAMERATE,
    AlignHorizontal,
    AlignVertical,
    Hud,
)
from tello_aruco_nav.modules.mission_controller import MissionController
from tello_aruco_nav.modules.tello import Tello, TelloConnectionState
from tello_aruco_nav.modules.tello_controller import TelloController, TelloState
from tello_aruco_nav.modules.ui import Ui
from tello_aruco_nav.schemas.calibration import CalibrationData
from tello_aruco_nav.schemas.map import MapData

logger = logging.getLogger("flight")


UPDATE_TIME = 0.05
RECONNECT_TIME = 5.0


def run_flight(
    map_file: str = "map.json",
    mission_file: str | None = None,
    calibration_file: str = "calibration.json",
):
    FlightRunner(map_file, mission_file, calibration_file).run()


class FlightRunner:
    def __init__(self, map_file: str, mission_file: str | None, calibration_file: str):
        calibration_data = load_json(CalibrationData, calibration_file)
        map_data = load_json(MapData, map_file)

        self.__tello = Tello()
        self.__camera = TelloCamera(self.__tello)
        self.__localizer = ArucoLocalization(
            map_data.markers,
            calibration_data.get_np_matrix(),
            calibration_data.get_np_dist_coeffs(),
            calibration_data.rotation,
            calibration_data.offset,
        )
        self.__controller = TelloController(
            self.__tello,
            calibration_data.pid_x,
            calibration_data.pid_y,
            calibration_data.pid_z,
            calibration_data.rates,
        )
        self.__mission_controller = MissionController(
            mission_file, map_data.markers, self.__controller
        )
        self.__hud = Hud()
        self.__flight_controller = FlightController(
            map_data.markers, self.__tello, self.__mission_controller, self.__controller
        )
        self.__ui = Ui(
            self.__tello,
            self.__controller,
            self.__mission_controller,
            self.__flight_controller,
            self.__hud,
        )

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

        self.__ui.stop()
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
        asyncio.create_task(self.__hud_loop())
        asyncio.create_task(self.__flight_controller.run(self.__ui))
        self.__ui.start()

        while self.__is_running:
            try:
                await self.__tello.connect()
                await self.__tello.stream_on()

                try:
                    controller_task = asyncio.create_task(self.__controller.run())
                    await self.__flight_loop()
                except (TelloDisconnectedException, TelloErrorException):
                    logger.error(f"Reconnecting in {RECONNECT_TIME} s.")
                finally:
                    self.__camera_img = None
                    self.__camera_gray_img = None
                    self.__controller.state = TelloState.IDLE
                    self.__mission_controller.reset()
                    controller_task.cancel()
            except (
                TelloFailedConnectException,
                TelloDisconnectedException,
                TelloErrorException,
            ):
                logger.error(f"Retrying in {RECONNECT_TIME} s.")

            await asyncio.sleep(RECONNECT_TIME)

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
            await asyncio.sleep(UPDATE_TIME)

    async def __hud_loop(self):
        last_frame_time = time()
        frame_delay = 1.0 / HUD_FRAMERATE
        while self.__is_running:
            last_frame_time = time()
            if not self.__ui.is_running:
                self.stop()
                break
            self.__hud.push_image(self.__camera_img)

            if self.__aruco_pos is not None:
                self.__hud.push_text(
                    f"x={self.__aruco_pos[0]:.2f}, y={self.__aruco_pos[1]:.2f}, z={self.__aruco_pos[2]:.2f}",
                    AlignHorizontal.LEFT,
                    AlignVertical.TOP,
                )
            if self.__aruco_rot is not None:
                euler = euler_from_matrix(self.__aruco_rot)
                self.__hud.push_text(
                    f"pitch={euler[0]:.0f}, yaw={euler[1]:.0f}, roll={euler[2]:.0f}",
                    AlignHorizontal.LEFT,
                    AlignVertical.TOP,
                )

            self.__hud.push_text(
                f"bat={self.__tello.battery:.0f}%",
                AlignHorizontal.RIGHT,
                AlignVertical.TOP,
            )
            self.__hud.push_text(
                f"h={self.__tello.height} m.", AlignHorizontal.RIGHT, AlignVertical.TOP
            )

            match self.__controller.state:
                case TelloState.IDLE:
                    self.__hud.push_text(
                        "idle", AlignHorizontal.CENTER, AlignVertical.TOP
                    )
                case TelloState.TAKEOFF:
                    self.__hud.push_text(
                        "taking off", AlignHorizontal.CENTER, AlignVertical.TOP
                    )
                case TelloState.GO_TO_POS:
                    target_pos = self.__controller.target_pos
                    dist = self.__controller.marker_dist
                    alt_delta = self.__controller.marker_alt_delta
                    if target_pos is not None:
                        self.__hud.push_text(
                            f"flying to x={target_pos[0]:.2f} y={target_pos[1]:.2f} z={target_pos[2]:.2f}",
                            AlignHorizontal.CENTER,
                            AlignVertical.TOP,
                        )
                    if dist is not None:
                        self.__hud.push_text(
                            f"dist={dist:.2f} m., delta_alt={alt_delta:.2f} m.",
                            AlignHorizontal.CENTER,
                            AlignVertical.TOP,
                        )
                case TelloState.MANUAL_CONTROL:
                    control = self.__controller.manual_control
                    self.__hud.push_text(
                        "manual", AlignHorizontal.CENTER, AlignVertical.TOP
                    )
                    if control is not None:
                        self.__hud.push_text(
                            f"x={control[0]}, y={control[2]}, z={control[1]}",
                            AlignHorizontal.CENTER,
                            AlignVertical.TOP,
                        )
                case TelloState.WAITING:
                    self.__hud.push_text(
                        "waiting", AlignHorizontal.CENTER, AlignVertical.TOP
                    )
                case TelloState.LANDING:
                    self.__hud.push_text(
                        "landing", AlignHorizontal.CENTER, AlignVertical.TOP
                    )

            match self.__flight_controller.mode:
                case FlightMode.MANUAL:
                    self.__hud.push_text(
                        "manual", AlignHorizontal.CENTER, AlignVertical.BOTTOM
                    )
                case FlightMode.FOLLOW:
                    marker_id = self.__flight_controller.target_marker_id
                    if marker_id is not None:
                        self.__hud.push_text(
                            f"following marker with id={marker_id}",
                            AlignHorizontal.CENTER,
                            AlignVertical.BOTTOM,
                        )
                case FlightMode.MISSION:
                    wp_idx = self.__mission_controller.waypoint_index
                    wp_cnt = self.__mission_controller.waypoints_count

                    if wp_idx is not None:
                        self.__hud.push_text(
                            f"running mission ({wp_idx + 1}/{wp_cnt})",
                            AlignHorizontal.CENTER,
                            AlignVertical.BOTTOM,
                        )
                    else:
                        self.__hud.push_text(
                            "running mission",
                            AlignHorizontal.CENTER,
                            AlignVertical.BOTTOM,
                        )

            self.__hud.update()
            await asyncio.sleep(max(frame_delay - time() + last_frame_time, 0.0))
