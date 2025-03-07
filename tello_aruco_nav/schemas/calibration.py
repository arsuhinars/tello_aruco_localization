from typing import Annotated

import numpy as np
from annotated_types import Len
from pydantic import BaseModel

from tello_aruco_nav.common.utils import Float3


class CalibrationData(BaseModel):
    matrix: Annotated[list[Annotated[list[float], Len(3, 3)]], Len(3, 3)]
    dist_coeffs: Annotated[list[float], Len(5, 5)]
    rotation: Float3

    pid_x: Float3
    pid_y: Float3
    pid_z: Float3

    def get_np_matrix(self):
        return np.array(self.matrix, np.float32)

    def get_np_dist_coeffs(self):
        return np.array(self.dist_coeffs, np.float32).reshape(-1, 1)
