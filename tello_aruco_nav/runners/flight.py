import signal

import pygame as pg

from tello_aruco_nav.common.utils import console, euler_from_matrix, load_json
from tello_aruco_nav.modules.aruco_localization import ArucoLocalization
from tello_aruco_nav.modules.camera import TelloCamera
from tello_aruco_nav.modules.drone_controller import DroneController, DroneState
from tello_aruco_nav.modules.gui import AlignHorizontal, AlignVertical, Gui
from tello_aruco_nav.modules.tello import Tello, TelloConnectionState
from tello_aruco_nav.schemas.calibration import CalibrationData
from tello_aruco_nav.schemas.map import MapData
from tello_aruco_nav.schemas.mission import MissionData


def run_flight(
    map_file: str = "map.json",
    mission_file: str | None = None,
    calibration_file: str = "camera.json",
):
    mission_data = (
        load_json(MissionData, mission_file)
        if mission_file is not None
        else MissionData(waypoints=[])
    )
    calibration_data = load_json(CalibrationData, calibration_file)
    map_data = load_json(MapData, map_file)

    tello = Tello()
    camera = TelloCamera(tello)
    localizer = ArucoLocalization(
        map_data.markers,
        calibration_data.get_np_matrix(),
        calibration_data.get_np_dist_coeffs(),
        calibration_data.rotation,
    )
    controller = DroneController(tello, mission_data, map_data.markers)
    gui = Gui()

    tello.connect()
    gui.run()

    def stop(signum, frame):
        gui.stop()
        console.log("Stopping app")
        if tello is not None and tello.is_flying:
            tello.emergency()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    console.log("App is running")

    while gui.is_running:
        tello.update()
        match tello.connection_state:
            case TelloConnectionState.DISCONNECTED:
                gui.stop()
            case TelloConnectionState.CONNECTED:
                if not tello.is_streaming:
                    tello.stream_on()

                if gui.is_key_just_pressed(pg.K_SPACE):
                    if controller.state == DroneState.IDLE:
                        controller.start()
                    else:
                        controller.stop()

                if gui.is_key_just_pressed(pg.K_ESCAPE):
                    tello.emergency()

        img = camera.read_image()
        gray, pos, rot_mtx = localizer.update(img)
        controller.update(pos, rot_mtx)
        gui.push_image(img)

        if pos is not None:
            gui.push_text(
                f"x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}",
                AlignHorizontal.LEFT,
                AlignVertical.TOP,
            )
        if rot_mtx is not None:
            rot = euler_from_matrix(rot_mtx)
            gui.push_text(
                f"pitch={rot[0]:.0f}, yaw={rot[1]:.0f}, roll={rot[2]:.0f}",
                AlignHorizontal.LEFT,
                AlignVertical.TOP,
            )

        gui.push_text(
            f"bat={tello.battery:.0f}%", AlignHorizontal.RIGHT, AlignVertical.TOP
        )
        gui.push_text(f"h={tello.height}", AlignHorizontal.RIGHT, AlignVertical.TOP)

        match controller.state:
            case DroneState.IDLE:
                gui.push_text(
                    "idle",
                    AlignHorizontal.CENTER,
                    AlignVertical.BOTTOM,
                )
            case DroneState.TAKING_OFF:
                gui.push_text(
                    "taking off",
                    AlignHorizontal.CENTER,
                    AlignVertical.BOTTOM,
                )
            case DroneState.FLYING:
                gui.push_text(
                    f"flying to wp {controller.waypoint_index} with id={controller.current_marker.id}",
                    AlignHorizontal.CENTER,
                    AlignVertical.BOTTOM,
                )
                gui.push_text(
                    f"dist={controller.waypoint_distance:.2f}, delta_alt={controller.waypoint_altitude_delta:.2f}",
                    AlignHorizontal.CENTER,
                    AlignVertical.BOTTOM,
                )
            case DroneState.WAITING:
                gui.push_text(
                    f"waiting in wp {controller.waypoint_index} with id={controller.current_marker.id}",
                    AlignHorizontal.CENTER,
                    AlignVertical.BOTTOM,
                )
                gui.push_text(
                    f"{controller.waypoint_wait_time:.1f} s. remained",
                    AlignHorizontal.CENTER,
                    AlignVertical.BOTTOM,
                )
                gui.push_text(
                    f"dist={controller.waypoint_distance:.2f}, delta_alt={controller.waypoint_altitude_delta:.2f}",
                    AlignHorizontal.CENTER,
                    AlignVertical.BOTTOM,
                )
            case DroneState.LANDING:
                gui.push_text(
                    "landing",
                    AlignHorizontal.CENTER,
                    AlignVertical.BOTTOM,
                )

        gui.update()

    camera.release()

    tello.stream_off()
    tello.disconnect()

    console.log("App was stopped")
