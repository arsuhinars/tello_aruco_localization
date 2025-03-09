import asyncio

from tello_aruco_nav.common.utils import load_json
from tello_aruco_nav.modules.tello_controller import TelloController, TelloState
from tello_aruco_nav.schemas.mission import MissionData

UPDATE_DELAY = 0.2


class MissionController:
    def __init__(self, mission_file: str | None, controller: TelloController):
        self.__waypoints = (
            load_json(MissionData, mission_file).waypoints
            if mission_file is not None
            else []
        )

        self.__controller = controller
        self.__is_started = False
        self.__curr_wp_idx: int | None = None

    @property
    def is_started(self):
        return self.__is_started

    @property
    def waypoint_index(self):
        return self.__curr_wp_idx

    @property
    def waypoints_count(self):
        return len(self.__waypoints)

    def start(self):
        self.__is_started = True
        self.__task = asyncio.create_task(self.__run())

    def stop(self):
        if self.__is_started:
            self.__task.cancel()
            if self.__controller.is_flying:
                self.__controller.state = TelloState.LANDING
            self.__is_started = False

    def reset(self):
        if self.__is_started:
            self.__task.cancel()
            self.__is_started = False

    async def __run(self):
        self.__is_started = True
        self.__curr_wp_idx = None
        self.__controller.state = TelloState.TAKEOFF
        await self.__controller.on_take_off()

        for i, wp in enumerate(self.__waypoints):
            self.__curr_wp_idx = i
            self.__controller.target_marker_id = wp.marker_id
            self.__controller.target_altitude = wp.altitude
            dist = self.__controller.marker_dist
            while dist is None or dist >= wp.radius:
                await asyncio.sleep(UPDATE_DELAY)
            await asyncio.sleep(wp.delay_after)

        self.__curr_wp_idx = None
        self.__controller.state = TelloState.LANDING
        await self.__controller.on_landed()
        self.__is_started = False
