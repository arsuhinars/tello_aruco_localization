from pydantic import BaseModel


class MissionWaypoint(BaseModel):
    marker_id: int
    altitude: float
    radius: float = 0.3
    delay_after: float = 0.0


class MissionData(BaseModel):
    waypoints: list[MissionWaypoint]
