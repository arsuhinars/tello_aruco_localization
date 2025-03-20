import cv2
import numpy as np
from cv2 import aruco
from cv2.typing import MatLike

from tello_aruco_nav.common.utils import ARUCO_DICTIONARY, Float3, rotation_matrix_euler
from tello_aruco_nav.schemas.map import MarkerData


class ArucoLocalization:
    def __init__(
        self,
        markers: list[MarkerData],
        camera_matrix: np.ndarray,
        camera_dist_coeffs: np.ndarray,
        camera_angles: Float3,
        camera_offset: Float3,
    ):
        self.__cam_mtx = camera_matrix
        self.__cam_dist = camera_dist_coeffs
        self.__cam_rotation_mtx = np.linalg.inv(rotation_matrix_euler(*camera_angles))[
            :3, :3
        ]
        self.__cam_offset = np.array(camera_offset, np.float32)

        dictionary = aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
        markers_ids = np.fromiter(map(lambda m: m.id, markers), np.int32)
        object_points = list(map(MarkerData.get_object_points, markers))
        self.__board = aruco.Board(object_points, dictionary, markers_ids)

        parameters = aruco.DetectorParameters()
        self.__detector = aruco.ArucoDetector(dictionary, parameters)

        self.__gray_img: np.ndarray | None = None

    def update(self, img: MatLike | None):
        if img is None:
            return None, None, None

        self.__gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, self.__gray_img)
        self.__gray_img = cv2.GaussianBlur(self.__gray_img, (3, 3), 0, self.__gray_img)
        # self.__gray_img = cv2.adaptiveThreshold(
        #     self.__gray_img,
        #     255.0,
        #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #     cv2.THRESH_BINARY,
        #     11,
        #     2.0,
        #     self.__gray_img,
        # )
        # _, gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        corners, ids, rejected_corners = self.__detector.detectMarkers(self.__gray_img)
        corners, ids, rejected_corners, _ = self.__detector.refineDetectedMarkers(
            self.__gray_img,
            self.__board,
            corners,
            ids,
            rejected_corners,
            self.__cam_mtx,
            self.__cam_dist,
        )
        aruco.drawDetectedMarkers(img, corners, ids)

        if ids is None or len(ids) == 0:
            return self.__gray_img, None, None

        object_points, image_points = self.__board.matchImagePoints(corners, ids)
        if (
            object_points is None
            or len(object_points) == 0
            or image_points is None
            or len(image_points) == 0
        ):
            return self.__gray_img, None, None

        result, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.__cam_mtx,
            self.__cam_dist,
        )

        if not result:
            return self.__gray_img, None, None

        rot, _ = cv2.Rodrigues(rvec)
        pos = -(rot.T @ tvec)
        rot = self.__cam_rotation_mtx @ rot
        rot_t = rot.T
        pos = pos.flatten() + self.__cam_offset @ rot_t

        cv2.drawFrameAxes(img, self.__cam_mtx, self.__cam_dist, rvec, tvec, 1.0)

        return self.__gray_img, pos.flatten(), rot_t
