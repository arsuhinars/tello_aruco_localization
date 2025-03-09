import signal
from typing import cast

import cv2
import numpy as np

from tello_aruco_nav.common.utils import console
from tello_aruco_nav.modules.camera import BaseCamera, CvCamera, TelloCamera
from tello_aruco_nav.modules.hud import AlignHorizontal, AlignVertical, Hud
from tello_aruco_nav.modules.tello import Tello, TelloConnectionState

CHESSBOARD_SIZE = (9, 6)


def calibrate_camera(offline: bool = False, offline_camera_index: int = 0):
    camera: BaseCamera
    if offline:
        tello = None
        camera = CvCamera(offline_camera_index)
    else:
        tello = Tello()
        tello.connect()
        camera = TelloCamera(tello)

    gui = Hud()
    gui.run()
    frame_counter = 0
    object_points = np.zeros(
        (CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3),
        np.float32,
    )
    object_points[:, :2] = np.mgrid[
        0 : CHESSBOARD_SIZE[0], 0 : CHESSBOARD_SIZE[1]
    ].T.reshape(-1, 2)
    image_points: list[np.ndarray] = []
    gray_img: np.ndarray | None = None

    def stop(signum, frame):
        gui.stop()
        console.log("Stopping app")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    console.log("App is running")

    while gui.is_running:
        if tello is not None:
            match tello.connection_state:
                case TelloConnectionState.DISCONNECTED:
                    gui.stop()
                case TelloConnectionState.CONNECTING:
                    gui.update()
                    return
                case TelloConnectionState.CONNECTED:
                    if not tello.is_streaming:
                        tello.stream_on()

        img = camera.read_image()
        if img is None:
            gui.update()
            return
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, gray_img)

        result, corners = cv2.findChessboardCorners(gray_img, CHESSBOARD_SIZE)
        if result:
            corners = cv2.cornerSubPix(
                gray_img,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001),
            )
            image_points.append(corners)

            cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, True)

            frame_counter += 1

        gui.push_image(img)
        gui.push_text(
            f"{frame_counter} frames captured",
            AlignHorizontal.LEFT,
            AlignVertical.TOP,
        )
        gui.update()

    console.log("Calculating calibration parameters...")

    assert gray_img is not None
    _, camera_matrix, camera_dist_coeffs, _, _ = cv2.calibrateCamera(
        [object_points] * len(image_points),
        image_points,
        cast(list[int], gray_img.shape[::-1]),
        np.zeros((3, 3), np.float32),
        np.zeros(14, np.float32),
    )

    np.set_printoptions(precision=4)
    console.print("matrix\n", camera_matrix, sep="")
    np.set_printoptions(precision=4)
    console.print("dist_coeffs\n", camera_dist_coeffs, sep="")

    camera.release()

    if tello is not None:
        tello.stream_off()
        tello.disconnect()

    console.log("App was stopped")
