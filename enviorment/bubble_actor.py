from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent_interface import NeighborhoodVehicles, RGB, OGM, DrivableAreaGridMap, Waypoints, RoadWaypoints, RoadWaypoints, DoneCriteria
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.registry import register
from smarts.sstudio.types import Bubble, MapZone, SocialAgentActor
from smarts.core.agent import Agent
from smarts.core.observations import Observation

import hyper_param as hp

from typing import Dict, Tuple
class DummyAgent(Agent):
    def act(self, obs: Observation) -> Tuple[float, float]:
        velocity = 10
        lane_change = 0.0
        return (velocity, lane_change)
social_interface = AgentInterface(action=ActionSpaceType.LaneWithContinuousSpeed,)
register(
    "Blank-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=social_interface,
        agent_builder=DummyAgent,
    ),
)

import importlib
def entry_point_IAMP(**kwargs):
    pkg = "interaction_aware_motion_prediction"
    module = ".policy"
    lib = importlib.import_module(module, pkg)

    return AgentSpec(
        interface=AgentInterface(
            waypoint_paths=Waypoints(hp.waypoints_num),
            neighborhood_vehicle_states=NeighborhoodVehicles(radius=30),
            action=ActionSpaceType.TargetPose,
            done_criteria=DoneCriteria(off_route=False,wrong_way=False,),
        ),
        agent_builder=lib.Policy,
    )
register(
    locator="IAMP-v1", entry_point=entry_point_IAMP
)

from agent.IDM_MOBIL import IDM_MOBIL
def entry_point_IDM_MOBIL(**kwargs):
    return AgentSpec(
        interface=AgentInterface(
            road_waypoints=RoadWaypoints(hp.waypoints_num),
            waypoint_paths=Waypoints(hp.waypoints_num),
            neighborhood_vehicle_states=NeighborhoodVehicles(radius=30),
            action=ActionSpaceType.LaneWithContinuousSpeed,
            done_criteria=DoneCriteria(off_route=False,wrong_way=False,),
        ),
        agent_builder=IDM_MOBIL,
    )
register(
    locator="IDM_MOBIL-v1", entry_point=entry_point_IDM_MOBIL
)

bubble_actor = SocialAgentActor(name="bubble actor", 
                                agent_locator="IDM_MOBIL-v1")