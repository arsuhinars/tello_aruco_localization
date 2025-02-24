import numpy as np
from rich.console import Console

console = Console()


def translation_matrix(x: float, y: float, z: float):
    return np.array(
        [
            [0.0, 0.0, 0.0, x],
            [0.0, 0.0, 0.0, y],
            [0.0, 0.0, 0.0, z],
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
