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
    def __init__(self, aruco_dict_id: int, map_data: MapData):
        dictionary = aruco.getPredefinedDictionary(aruco_dict_id)
        markers = map_data.convert_aruco_list()
        markers_ids = np.array(map(lambda m: m.id, markers), np.int32)
        object_points = list(map(ArucoCenter.get_object_points, markers))
        self.__board = aruco.Board(
            [i for j in object_points for i in j], dictionary, markers_ids
        )

        parameters = aruco.DetectorParameters()
        self.__detector = aruco.ArucoDetector(dictionary, parameters)

    def update(self, image: MatLike):
        corners, ids, rejected_points = self.__detector.detectMarkers(image)

        aruco.drawDetectedMarkers(image, corners, ids)
