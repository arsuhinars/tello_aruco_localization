from pydantic import BaseModel

from tello_aruco_nav.common.utils import Float3


class MissionMarkerWaypoint(BaseModel):
    marker_id: int
    altitude: float
    radius: float = 0.3
    delay_after: float = 0.0


class MissionLocationWaypoint(BaseModel):
    position: Float3
    radius: float = 0.3
    delay_after: float = 0.0


class MissionFlyOffsetWaypoint(BaseModel):
    offset: Float3
    delay_after: float = 0.0


class MissionRcControlWaypoint(BaseModel):
    control: tuple[int, int, int]
    duration: float
    delay_after: float = 0.0


class MissionData(BaseModel):
    waypoints: list[
        MissionLocationWaypoint
        | MissionMarkerWaypoint
        | MissionFlyOffsetWaypoint
        | MissionRcControlWaypoint
    ]
