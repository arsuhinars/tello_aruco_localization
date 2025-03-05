import signal

import cv2
import pygame as pg
from djitellopy import Tello

from tello_aruco_nav.common.utils import console, euler_from_matrix, load_json
from tello_aruco_nav.modules.aruco_localization import ArucoLocalization
from tello_aruco_nav.modules.camera import BaseCamera, CvCamera, TelloCamera
from tello_aruco_nav.modules.drone_controller import DroneController
from tello_aruco_nav.modules.gui import AlignHorizontal, AlignVertical, Gui
from tello_aruco_nav.schemas.calibration import CalibrationData
from tello_aruco_nav.schemas.map import MapData
from tello_aruco_nav.schemas.mission import MissionData


def run_flight(
    map_file: str = "map.json",
    mission_file: str | None = None,
    calibration_file: str = "camera.json",
    offline: bool = False,
    offline_camera_index: int = 0,
):
    mission_data = (
        load_json(MissionData, mission_file)
        if mission_file is not None
        else MissionData(waypoints=[])
    )
    calibration_data = load_json(CalibrationData, calibration_file)
    map_data = load_json(MapData, map_file)

    camera: BaseCamera
    if offline:
        tello = None
        camera = CvCamera(offline_camera_index)
    else:
        tello = Tello()
        tello.connect()
        tello.streamon()
        camera = TelloCamera(tello)

    localizer = ArucoLocalization(
        map_data.markers,
        calibration_data.get_np_matrix(),
        calibration_data.get_np_dist_coeffs(),
        calibration_data.rotation,
    )
    controller = (
        DroneController(tello, mission_data, map_data.markers)
        if tello is not None and mission_data is not None
        else None
    )
    gui = Gui()
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
        img = camera.read_image()
        gray, pos, rot_mtx = localizer.update(img)

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

        if tello is not None:
            speed = (
                tello.get_speed_x() / 100.0,
                tello.get_speed_y() / 100.0,
                tello.get_speed_z() / 100.0,
            )
            cv2.putText(
                img,
                f"speed = {speed[0]:.2f}, {speed[1]:.2f}, {speed[2]:.2f}",
                (20, 60),
                cv2.FONT_HERSHEY_PLAIN,
                1.0,
                WHITE_COLOR,
                1,
            )
            cv2.putText(
                img,
                f"battery = {tello.get_battery()}%",
                (20, 80),
                cv2.FONT_HERSHEY_PLAIN,
                1.0,
                WHITE_COLOR,
                1,
            )

        window_width, window_height = pg.display.get_window_size()
        if img is not None and (
            img.shape[1] != window_width or img.shape[0] != window_height
        ):
            pg.display.set_mode((img.shape[1], img.shape[0]))

        if img is not None and img.size != 0:
            pg.surfarray.blit_array(surface, img.transpose(1, 0, 2))
        pg.display.flip()

        if controller is not None:
            controller.update(pos, rot_mtx)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                is_running = False

    camera.release()

    if tello is not None:
        tello.streamoff()
        tello.end()

    pg.quit()

    console.log("App was stopped")
