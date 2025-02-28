import itertools

import numpy as np
from cv2 import aruco
from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from pydantic_settings import BaseSettings, JsonConfigSettingsSource, SettingsConfigDict

from tello_aruco_nav.utils import (
    Float3,
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    translation_matrix,
)


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(json_file="settings.json")

    camera_matrix: list[list[float]] | None = None
    camera_dist_coeffs: list[float] | None = None
    camera_angles: Float3 | None = None

    aruco_dictionary: int = aruco.DICT_4X4_250

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (JsonConfigSettingsSource(settings_cls),)

    def convert_camera_matrix(self):
        return np.array(self.camera_matrix, np.float32)

    def convert_camera_dist_coeffs(self):
        return np.array(self.camera_dist_coeffs, np.float32).reshape(-1, 1)


@dataclass
class ArucoCenter:
    id: int
    center: Float3
    rotation: Float3
    size: float

    def get_object_points(self) -> np.ndarray:
        m = (
            translation_matrix(*self.center)
            @ rotation_matrix_y(np.deg2rad(self.rotation[1]))
            @ rotation_matrix_x(np.deg2rad(self.rotation[0]))
            @ rotation_matrix_z(np.deg2rad(self.rotation[2]))
        )

        points = np.array(
            [
                [-self.size, 0.0, self.size, 1.0],
                [self.size, 0.0, self.size, 1.0],
                [self.size, 0.0, -self.size, 1.0],
                [-self.size, 0.0, -self.size, 1.0],
            ],
            np.float32,
        )
        np.dot(points, m.T, points)

        return points[:, :-1]


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


class MissionWaypoint(BaseModel):
    marker_id: int
    altitude: float
    radius: float = 0.3
    delay_after: float = 0.0


class MissionData(BaseModel):
    waypoints: list[MissionWaypoint]
