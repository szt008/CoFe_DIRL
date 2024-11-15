from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent_interface import NeighborhoodVehicles, RGB, OGM, DrivableAreaGridMap, Waypoints
from smarts.core.agent_interface import (
    OGM,
    RGB,
    AgentInterface,
    DoneCriteria,
    DrivableAreaGridMap,
    RoadWaypoints,
    Waypoints,
)
from smarts.core.controllers import ActionSpaceType, ControllerOutOfLaneException
import hyper_param as hp

agent_interface = AgentInterface(
    road_waypoints=RoadWaypoints(hp.waypoints_num), 
    waypoint_paths=Waypoints(hp.waypoints_num),
    neighborhood_vehicle_states=NeighborhoodVehicles(radius=hp.vehicle_obs_distance),
    top_down_rgb=RGB(hp.screen_width, hp.screen_height,  hp.view/hp.screen_height),
    occupancy_grid_map=OGM(hp.screen_height, hp.screen_width,  hp.view/hp.screen_height),
    drivable_area_grid_map=DrivableAreaGridMap(hp.screen_height, hp.screen_width,  hp.view/hp.screen_height),
    action=ActionSpaceType.LaneWithContinuousSpeed ,
)
