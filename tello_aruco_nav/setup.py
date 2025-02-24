import os
import signal
from typing import cast

import cv2
import numpy as np
import pygame as pg
import typer
from cv2 import aruco
from djitellopy import Tello

from tello_aruco_nav.aruco_localization import ArucoLocalization
from tello_aruco_nav.camera import BaseCamera, CvCamera, TelloCamera
from tello_aruco_nav.settings import AppSettings, MapData
from tello_aruco_nav.utils import console

CHESSBOARD_SIZE = (9, 6)

app = typer.Typer()


@app.command()
def generate_markers(ids: list[int], output_dir: str = "", size: int = 200):
    settings = AppSettings()
    dictionary = aruco.getPredefinedDictionary(settings.aruco_dictionary)
    img = np.zeros((size, size, 3), np.uint8)

    os.makedirs(output_dir, exist_ok=True)

    for id in ids:
        aruco.generateImageMarker(dictionary, id, size, img)
        cv2.imwrite(os.path.join(output_dir, f"marker_{id}.png"), img)


@app.command()
def calibrate_camera(offline: bool = False, offline_camera_index: int = 0):
    camera: BaseCamera
    if offline:
        tello = None
        camera = CvCamera(offline_camera_index)
    else:
        tello = Tello()
        tello.connect()
        tello.streamon()
        camera = TelloCamera(tello)

    pg.display.init()

    surface = pg.display.set_mode((640, 480))
    is_running = True
    frame_counter = 0
    object_points = np.zeros(
        (CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3),
        np.float32,
    )
    object_points[:, :2] = np.mgrid[
        0 : CHESSBOARD_SIZE[0], 0 : CHESSBOARD_SIZE[1]
    ].T.reshape(-1, 2)
    image_points: list[np.ndarray] = []

    def stop(signum, frame):
        nonlocal is_running
        is_running = False
        console.log("Stopping app")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    console.log("App is running")

    while is_running:
        img = camera.read_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        result, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE)
        if result:
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001),
            )
            image_points.append(corners)

            cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, True)

            frame_counter += 1

        cv2.putText(
            img,
            f"{frame_counter} frames captured",
            (10, 20),
            cv2.FONT_HERSHEY_PLAIN,
            1.0,
            (255.0, 255.0, 255.0),
        )

        window_width, window_height = pg.display.get_window_size()
        if img.shape[1] != window_width or img.shape[0] != window_height:
            pg.display.set_mode((img.shape[1], img.shape[0]))

        pg.surfarray.blit_array(surface, img.transpose(1, 0, 2))
        pg.display.flip()

        for event in pg.event.get():
            if event.type == pg.QUIT:
                is_running = False

    console.log("Calculating calibration parameters...")

    ret, camera_matrix, camera_dist_coeffs, _, _ = cv2.calibrateCamera(
        [object_points] * len(image_points),
        image_points,
        cast(list[int], gray.shape[::-1]),
        np.zeros((3, 3), np.float32),
        np.zeros(14, np.float32),
    )

    console.print("ret\n", ret, sep="")
    console.print("matrix\n", camera_matrix, sep="")
    console.print("dist_coeffs\n", camera_dist_coeffs, sep="")

    camera.release()

    if tello is not None:
        tello.streamoff()
        tello.end()

    pg.quit()

    console.log("App was stopped")


@app.command()
def run_localization(
    map_file: str, offline: bool = False, offline_camera_index: int = 0
):
    settings = AppSettings()

    camera: BaseCamera
    if offline:
        tello = None
        camera = CvCamera(offline_camera_index)
    else:
        tello = Tello()
        tello.connect()
        tello.streamon()
        camera = TelloCamera(tello)

    with open(map_file) as f:
        map_data = MapData.model_validate_json(f.read())
    localization = ArucoLocalization(settings.aruco_dictionary, map_data)

    pg.display.init()
    surface = pg.display.set_mode((640, 480))
    is_running = True

    def stop(signum, frame):
        nonlocal is_running
        is_running = False
        console.log("Stopping app")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    console.log("App is running")

    while is_running:
        img = camera.read_image()

        localization.update(img)

        pg.surfarray.blit_array(surface, img.transpose(1, 0, 2))
        pg.display.flip()

        for event in pg.event.get():
            if event.type == pg.QUIT:
                is_running = False

    camera.release()

    if tello is not None:
        tello.streamoff()
        tello.end()

    pg.quit()

    console.log("App was stopped")


@app.command()
def run_mission(map_file: str, mission_file: str): ...
