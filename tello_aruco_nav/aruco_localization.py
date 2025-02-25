import cv2
import numpy as np
from cv2 import aruco
from cv2.typing import MatLike

from tello_aruco_nav.settings import ArucoCenter, MapData


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

    def update(self, image: MatLike):
        corners, ids, rejected_corners = self.__detector.detectMarkers(image)
        corners, ids, rejected_corners, recovered_ids = (
            self.__detector.refineDetectedMarkers(
                image,
                self.__board,
                corners,
                ids,
                rejected_corners,
                self.__cam_mtx,
                self.__cam_dist,
            )
        )
        aruco.drawDetectedMarkers(image, corners, ids)

        if ids is None or len(ids) == 0:
            return

        object_points, image_points = self.__board.matchImagePoints(corners, ids)
        result, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.__cam_mtx,
            self.__cam_dist,
        )

        cv2.drawFrameAxes(image, self.__cam_mtx, self.__cam_dist, rvec, tvec, 1.0)
