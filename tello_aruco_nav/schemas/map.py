from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from tello_aruco_nav.common.utils import (
    Float3,
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    translation_matrix,
)


@dataclass
class MarkerData:
    id: int
    center: Float3
    size: float
    rotation: Float3 = (0.0, 0.0, 0.0)

    def get_object_points(self) -> np.ndarray:
        m = (
            translation_matrix(*self.center)
            @ rotation_matrix_y(np.deg2rad(self.rotation[1]))
            @ rotation_matrix_x(np.deg2rad(self.rotation[0]))
            @ rotation_matrix_z(np.deg2rad(self.rotation[2]))
        )

        half_size = self.size / 2.0
        points = np.array(
            [
                [-half_size, 0.0, half_size, 1.0],
                [half_size, 0.0, half_size, 1.0],
                [half_size, 0.0, -half_size, 1.0],
                [-half_size, 0.0, -half_size, 1.0],
            ],
            np.float32,
        )
        np.dot(points, m.T, points)

        return points[:, :-1]


class MapData(BaseModel):
    markers: list[MarkerData]
