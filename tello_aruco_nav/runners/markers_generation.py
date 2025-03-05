import os

import cv2
import numpy as np
from cv2 import aruco

from tello_aruco_nav.schemas.calibration import AppSettings


def generate_markers(ids: list[int], output_dir: str = "", size: int = 200):
    settings = AppSettings()
    dictionary = aruco.getPredefinedDictionary(settings.aruco_dictionary)
    img = np.zeros((size, size, 3), np.uint8)

    os.makedirs(output_dir, exist_ok=True)

    for id in ids:
        aruco.generateImageMarker(dictionary, id, size, img)
        cv2.imwrite(os.path.join(output_dir, f"marker_{id}.png"), img)
