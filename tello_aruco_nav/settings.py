import itertools
from typing import TypeAlias

import numpy as np
from cv2 import aruco
from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from pydantic_settings import BaseSettings, SettingsConfigDict

from tello_aruco_nav.utils import (
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    translation_matrix,
)

Float3: TypeAlias = tuple[float, float, float]


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(json_file="settings.json")

    camera_matrix: list[float] | None = None
    camera_dist_coeffs: list[float] | None = None

    aruco_dictionary: int = aruco.DICT_6X6_250


@dataclass
class ArucoCenter:
    id: int
    center: Float3
    rotation: Float3
    size: float

    def get_object_points(self) -> list[np.ndarray]:
        m = (
            translation_matrix(*self.center)
            @ rotation_matrix_y(self.rotation[1])
            @ rotation_matrix_x(self.rotation[0])
            @ rotation_matrix_z(self.rotation[2])
        )

        points = np.array(
            [
                [self.size, self.size, 0.0, 1.0],
                [-self.size, self.size, 0.0, 1.0],
                [-self.size, -self.size, 0.0, 1.0],
                [self.size, -self.size, 0.0, 1.0],
            ],
            np.float32,
        )
        np.dot(points, m.T, points)

        return np.split(points[:, :, :-1], points.shape[0], axis=0)


@dataclass
class ArucoGrid:
    size: tuple[int, int]
    ids: list[int]
    marker_size: float
    marker_gap: float
    start: Float3 = (0.0, 0.0, 0.0)


class MapData(BaseModel):
    aruco: list[ArucoGrid | ArucoCenter]

    def convert_aruco_list(self) -> list[ArucoCenter]:
        result: list[ArucoCenter] = []

        for a in self.aruco:
            match a:
                case ArucoCenter():
                    result.append(a)
                case ArucoGrid():
                    for id, (y, x) in zip(
                        a.ids,
                        itertools.product(range(a.size[1]), range(a.size[0])),
                    ):
                        result.append(
                            ArucoCenter(
                                id,
                                (
                                    a.start[0] + (a.marker_size + a.marker_gap) * x,
                                    a.start[1] + (a.marker_size + a.marker_gap) * y,
                                    a.start[2],
                                ),
                                (0.0, 0.0, 0.0),
                                a.marker_size,
                            )
                        )

        return result
