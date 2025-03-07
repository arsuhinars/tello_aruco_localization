import os

import cv2
import numpy as np
from cv2 import aruco

from tello_aruco_nav.common.utils import ARUCO_DICTIONARY


def generate_markers(ids: list[int], output_dir: str = "", size: int = 200):
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    img = np.zeros((size, size, 3), np.uint8)

    os.makedirs(output_dir, exist_ok=True)

    for id in ids:
        aruco.generateImageMarker(dictionary, id, size, img)
        cv2.imwrite(os.path.join(output_dir, f"marker_{id}.png"), img)
