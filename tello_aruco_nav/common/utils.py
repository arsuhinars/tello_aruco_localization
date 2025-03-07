from typing import Type, TypeAlias

import numpy as np
from cv2 import aruco
from pydantic import BaseModel
from rich.console import Console

console = Console()

ARUCO_DICTIONARY = aruco.DICT_4X4_250

Float3: TypeAlias = tuple[float, float, float]


def load_json[T: BaseModel](model: Type[T], file_path: str) -> T:
    with open(file_path, mode="r") as f:
        return model.model_validate_json(f.read())


def translation_matrix(x: float, y: float, z: float):
    return np.array(
        [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ],
        np.float32,
    )


def rotation_matrix_x(angle: float):
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle), 0.0],
            [0.0, np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        np.float32,
    )


def rotation_matrix_y(angle: float):
    return np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        np.float32,
    )


def rotation_matrix_z(angle: float):
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0, 0.0],
            [np.sin(angle), np.cos(angle), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        np.float32,
    )


def rotation_matrix_euler(x: float, y: float, z: float):
    return (
        rotation_matrix_y(np.deg2rad(y))
        @ rotation_matrix_x(np.deg2rad(x))
        @ rotation_matrix_z(np.deg2rad(z))
    )


def euler_from_matrix(rotation_matrix: np.ndarray):
    sy = np.sqrt(
        rotation_matrix[0, 0] * rotation_matrix[0, 0]
        + rotation_matrix[1, 0] * rotation_matrix[1, 0]
    )
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = 0.0

    return np.array(
        [
            np.degrees(roll),
            np.degrees(pitch),
            np.degrees(yaw),
        ],
        np.float32,
    )
