import asyncio

from tello_aruco_nav.common.utils import load_json
from tello_aruco_nav.modules.tello_controller import TelloController
from tello_aruco_nav.schemas.mission import MissionData

UPDATE_DELAY = 0.2


class MissionController:
    def __init__(self, mission_file: str | None, controller: TelloController):
        self.__mission_file = mission_file
        self.reload_mission()

        self.__controller = controller
        self.__is_started = False
        self.__curr_wp_idx = 0

    def start(self): ...

    def stop(self): ...

    def reset(self): ...

    def reload_mission(self):
        self.__waypoints = (
            load_json(MissionData, self.__mission_file).waypoints
            if self.__mission_file is not None
            else []
        )

    async def run(self):
        while True:
            await asyncio.sleep(UPDATE_DELAY)
