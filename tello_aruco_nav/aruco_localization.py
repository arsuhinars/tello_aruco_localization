import cv2
import numpy as np
from cv2 import aruco
from cv2.typing import MatLike

from tello_aruco_nav.settings import ArucoCenter
from tello_aruco_nav.utils import Float3, rotation_matrix_euler


class ArucoResult:
    position: np.ndarray
    markers: list[np.ndarray]
    corners: list[np.ndarray]
    ids: np.ndarray
    rejected_points: list[np.ndarray]


class ArucoLocalization:
    def __init__(
        self,
        aruco_dict_id: int,
        markers: list[ArucoCenter],
        camera_matrix: np.ndarray,
        camera_dist_coeffs: np.ndarray,
        camera_angles: Float3,
    ):
        self.__cam_mtx = camera_matrix
        self.__cam_dist = camera_dist_coeffs
        self.__cam_rotation_mtx = np.linalg.inv(rotation_matrix_euler(*camera_angles))[
            :3, :3
        ]

        dictionary = aruco.getPredefinedDictionary(aruco_dict_id)
        markers_ids = np.fromiter(map(lambda m: m.id, markers), np.int32)
        object_points = list(map(ArucoCenter.get_object_points, markers))
        self.__board = aruco.Board(object_points, dictionary, markers_ids)

        parameters = aruco.DetectorParameters()
        self.__detector = aruco.ArucoDetector(dictionary, parameters)

    def update(self, img: MatLike):
        if img is None or img.size == 0:
            return None, None, None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        # _, gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        corners, ids, rejected_corners = self.__detector.detectMarkers(gray)
        corners, ids, rejected_corners, _ = self.__detector.refineDetectedMarkers(
            gray,
            self.__board,
            corners,
            ids,
            rejected_corners,
            self.__cam_mtx,
            self.__cam_dist,
        )
        aruco.drawDetectedMarkers(img, corners, ids)

        if ids is None or len(ids) == 0:
            return gray, None, None

        object_points, image_points = self.__board.matchImagePoints(corners, ids)
        if (
            object_points is None
            or len(object_points) == 0
            or image_points is None
            or len(image_points) == 0
        ):
            return None, None, None

        result, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.__cam_mtx,
            self.__cam_dist,
        )

        if not result:
            return None, None, None

        rot, _ = cv2.Rodrigues(rvec)
        pos = -(rot.T @ tvec)
        rot = self.__cam_rotation_mtx @ rot

        cv2.drawFrameAxes(img, self.__cam_mtx, self.__cam_dist, rvec, tvec, 1.0)

        return gray, pos.flatten(), rot
