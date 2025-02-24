import numpy as np
from rich.console import Console

console = Console()


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
        yaw = 0

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
