import cv2
import numpy as np
from cv2 import aruco
from cv2.typing import MatLike

from tello_aruco_nav.settings import ArucoCenter, MapData
from tello_aruco_nav.utils import euler_from_matrix


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
        map_data: MapData,
        camera_matrix: np.ndarray,
        camera_dist_coeffs: np.ndarray,
    ):
        self.__cam_mtx = camera_matrix
        self.__cam_dist = camera_dist_coeffs

        dictionary = aruco.getPredefinedDictionary(aruco_dict_id)
        markers = map_data.convert_aruco_list()
        markers_ids = np.fromiter(map(lambda m: m.id, markers), np.int32)
        object_points = list(map(ArucoCenter.get_object_points, markers))
        self.__board = aruco.Board(object_points, dictionary, markers_ids)

        parameters = aruco.DetectorParameters()
        self.__detector = aruco.ArucoDetector(dictionary, parameters)

    def update(self, img: MatLike):
        img_copy = img.copy()

        corners, ids, rejected_corners = self.__detector.detectMarkers(img)
        corners, ids, rejected_corners, recovered_ids = (
            self.__detector.refineDetectedMarkers(
                img,
                self.__board,
                corners,
                ids,
                rejected_corners,
                self.__cam_mtx,
                self.__cam_dist,
            )
        )
        aruco.drawDetectedMarkers(img_copy, corners, ids)

        if ids is None or len(ids) == 0:
            return img_copy

        object_points, image_points = self.__board.matchImagePoints(corners, ids)
        result, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.__cam_mtx,
            self.__cam_dist,
        )

        rotation, _ = cv2.Rodrigues(rvec)
        position = -(rotation.T @ tvec)

        pos_x, pos_y, pos_z = position[0, 0], position[1, 0], position[2, 0]
        pitch, yaw, roll = euler_from_matrix(rotation)

        cv2.drawFrameAxes(img_copy, self.__cam_mtx, self.__cam_dist, rvec, tvec, 1.0)
        cv2.putText(
            img_copy,
            f"pos = {pos_x:.2f}, {pos_y:.2f}, {pos_z:.2f}",
            (20, 20),
            cv2.FONT_HERSHEY_PLAIN,
            1.0,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            img_copy,
            f"angles = {pitch:.2f}, {yaw:.2f}, {roll:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_PLAIN,
            1.0,
            (255, 255, 255),
            1,
        )

        return img_copy
