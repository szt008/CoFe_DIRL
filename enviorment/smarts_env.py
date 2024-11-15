import logging
import os
from dataclasses import asdict, is_dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, SupportsFloat, Tuple, Union

import pandas as pd
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from gymnasium.envs.registration import EnvSpec

from envision import types as envision_types
from envision.client import Client as Envision
from envision.data_formatter import EnvisionDataFormatterArgs
from smarts.core import current_seed
from smarts.core import seed as smarts_seed
from smarts.core.agent_interface import AgentInterface
from smarts.core.local_traffic_provider import LocalTrafficProvider
from smarts.core.scenario import Scenario
from smarts.env.configs.hiway_env_configs import (
    EnvReturnMode,
    ScenarioOrder,
    SumoOptions,
)
from smarts.env.utils.action_conversion import ActionOptions, ActionSpacesFormatter
from smarts.env.utils.observation_conversion import (
    ObservationOptions,
    ObservationSpacesFormatter,
)
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.actor import ActorRole
from smarts.core.plan import Plan, PositionalGoal, Mission, Start, default_entry_tactic
from smarts.sstudio.types import EndlessMission, Route, TrafficActor, TrapEntryTactic
import os
import sys
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_PATH)
from common.print_color import printColor

DEFAULT_VISUALIZATION_CLIENT_BUILDER = partial(
    Envision,
    endpoint=None,
    output_dir=None,
    headless=False,
    data_formatter_args=EnvisionDataFormatterArgs("base", enable_reduction=False),
)

class SmartsNgsimEnv(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"],
    }
    """Metadata for gym's use."""

    # define render_mode if your environment supports rendering
    render_mode: Optional[str] = None
    reward_range = (-float("inf"), float("inf"))
    spec: Optional[EnvSpec] = None

    # Set these in ALL subclasses
    action_space: spaces.Space
    observation_space: spaces.Space

    # Created
    _np_random: Optional[np.random.Generator] = None

    def __init__(
        self,
        scenarios: Sequence[str],
        agent_interfaces: Dict[str, Union[Dict[str, Any], AgentInterface]],
        sim_name: Optional[str] = None,
        scenarios_order: ScenarioOrder = ScenarioOrder.default,
        headless: bool = False,
        visdom: bool = False,
        fixed_timestep_sec: float = 0.1,
        seed: int = 42,
        sumo_options: Union[Dict[str, Any], SumoOptions] = SumoOptions(),
        visualization_client_builder: partial = DEFAULT_VISUALIZATION_CLIENT_BUILDER,
        observation_options: Union[
            ObservationOptions, str
        ] = ObservationOptions.default,
        action_options: Union[ActionOptions, str] = ActionOptions.default,
        environment_return_mode: Union[EnvReturnMode, str] = EnvReturnMode.default,
        render_mode: Optional[str] = None,
    ):
        self._log = logging.getLogger(self.__class__.__name__)
        smarts_seed(seed)
        self._agent_interfaces: Dict[str, AgentInterface] = {
            a_id: (
                a_interface
                if isinstance(a_interface, AgentInterface)
                else AgentInterface(**a_interface)
            )
            for a_id, a_interface in agent_interfaces.items()
        }
        self._dones_registered = 0

        scenarios = [str(Path(scenario).resolve()) for scenario in scenarios]
        
        self._scenarios_iterator = Scenario.scenario_variations(
            scenarios,
            list(agent_interfaces.keys()),
            shuffle_scenarios=scenarios_order == ScenarioOrder.scrambled,
        )

        visualization_client = None
        if not headless:
            visualization_client = visualization_client_builder(
                headless=headless,
                sim_name=sim_name,
            )
            preamble = envision_types.Preamble(scenarios=scenarios)
            visualization_client.send(preamble)

        self._env_renderer = None
        self.render_mode = render_mode

        traffic_sims = []
        if Scenario.any_support_sumo_traffic(scenarios):
            if is_dataclass(sumo_options):
                sumo_options = asdict(sumo_options)
            sumo_traffic = SumoTrafficSimulation(
                headless=sumo_options["headless"],
                time_resolution=fixed_timestep_sec,
                num_external_sumo_clients=sumo_options["num_external_clients"],
                sumo_port=sumo_options["port"],
                auto_start=sumo_options["auto_start"],
            )
            traffic_sims += [sumo_traffic]
        smarts_traffic = LocalTrafficProvider()
        traffic_sims += [smarts_traffic]

        if isinstance(environment_return_mode, str):
            self._environment_return_mode = EnvReturnMode[environment_return_mode]
        else:
            self._environment_return_mode = environment_return_mode

        if isinstance(action_options, str):
            action_options = ActionOptions[action_options]
        self._action_formatter = ActionSpacesFormatter(
            agent_interfaces, action_options=action_options
        )
        self.action_space = self._action_formatter.space

        if isinstance(observation_options, str):
            observation_options = ObservationOptions[observation_options]
        self._observations_formatter = ObservationSpacesFormatter(
            agent_interfaces, observation_options
        )
        self.observation_space = self._observations_formatter.space

        from smarts.core.smarts import SMARTS

        self._smarts = SMARTS(
            agent_interfaces=agent_interfaces,
            traffic_sims=traffic_sims,
            envision=visualization_client,
            visdom=visdom,
            fixed_timestep_sec=fixed_timestep_sec,
        )

    def step(
        self, action: ActType
    ) -> Union[
        Tuple[Dict[str, Any], SupportsFloat, bool, bool, Dict[str, Any]],
        Tuple[
            Dict[str, Any],
            Dict[str, float],
            Dict[str, bool],
            Dict[str, bool],
            Dict[str, Any],
        ],
    ]:
        """
        Returns:
            (dict, SupportsFloat, bool, bool, dict):
                - observation. An element of the environment's :attr:`observation_space` as the
                    next observation due to the agent actions. This observation will change based on
                    the provided :attr:`agent_interfaces`. Check :attr:`observation_space` after
                    initialization.
                - reward. The reward as a result of taking the action.
                - terminated. Whether the agent reaches the terminal state (as defined under the MDP of the task)
                    which can be positive or negative. An example is reaching the goal state. If true, the user needs to call :meth:`reset`.
                - truncated. Whether the truncation condition outside the scope of the MDP is satisfied.
                    Typically, this is a time-limit, but could also be used to indicate an agent physically going out of bounds.
                    Can be used to end the episode prematurely before a terminal state is reached.
                    If true, the user needs to call :meth:`reset`.
                - info. Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                    This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                    hidden from observations, or individual reward terms that are combined to produce the total reward.
        """
        assert isinstance(action, dict) and all(
            isinstance(key, str) for key in action.keys()
        ), "Expected Dict[str, Any]"

        formatted_action = self._action_formatter.format(action)
        observations, rewards, dones, extras = self._smarts.step(formatted_action)

        info = {
            agent_id: {
                "score": agent_score,
                "env_obs": observations[agent_id],
                "done": dones[agent_id],
                "reward": rewards[agent_id],
                "map_source": self._smarts.scenario.road_map.source,
            }
            for agent_id, agent_score in extras["scores"].items()
        }

        if self._env_renderer is not None:
            self._env_renderer.step(observations, rewards, dones, info)

        for done in dones.values():
            self._dones_registered += 1 if done else 0

        dones["__all__"] = self._dones_registered >= len(self._agent_interfaces)

        assert all("score" in v for v in info.values())

        if self._environment_return_mode == EnvReturnMode.environment:
            return (
                self._observations_formatter.format(observations),
                sum(r for r in rewards.values()),
                dones["__all__"],
                dones["__all__"],
                info,
            )
        elif self._environment_return_mode == EnvReturnMode.per_agent:
            observations = self._observations_formatter.format(observations)
            if (
                self._observations_formatter.observation_options
                == ObservationOptions.full
            ):
                dones = {**{id_: False for id_ in observations}, **dones}
                return (
                    observations,
                    {**{id_: np.nan for id_ in observations}, **rewards},
                    dones,
                    dones,
                    info,
                )
            else:
                return (
                    observations,
                    rewards,
                    dones,
                    dones,
                    info,
                )
        raise RuntimeError(
            f"Invalid observation configuration using {self._environment_return_mode}"
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        options = options or {}
        scenario = options.get("scenario")
        
        if scenario is None:
            scenario = next(self._scenarios_iterator)
        
        self._dones_registered = 0
        observations = self._smarts.reset(
            scenario, start_time=options.get("start_time", 0)
        )
        info = {
            agent_id: {
                "score": 0,
                "env_obs": agent_obs,
                "done": False,
                "reward": 0,
                "map_source": self._smarts.scenario.road_map.source,
            }
            for agent_id, agent_obs in observations.items()
        }

        if self._env_renderer is not None:
            self._env_renderer.reset(observations)

        if seed is not None:
            smarts_seed(seed)
        return self._observations_formatter.format(observations), info

    def render(
        self,
    ) -> Optional[Union[gym.core.RenderFrame, List[gym.core.RenderFrame]]]:
        if self.render_mode == "rgb_array":
            if self._env_renderer is None:
                from smarts.env.utils.record import AgentCameraRGBRender

                self._env_renderer = AgentCameraRGBRender(self)

            return self._env_renderer.render(env=self)

    def close(self):
        if self._smarts is not None:
            self._smarts.destroy()

    @property
    def unwrapped(self) -> gym.Env[ObsType, ActType]:
        return self

    @property
    def np_random(self) -> np.random.Generator:
        return super().np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self._np_random = value

    def __str__(self):
        return super().__str__()

    def __enter__(self):
        """Support with-statement for the environment."""
        return super().__enter__()

    def __exit__(self, *args: Any):
        """Support with-statement for the environment and closes the environment."""
        self.close()
        # propagate exception
        return False

    @property
    def agent_ids(self) -> Set[str]:
        return set(self._agent_interfaces)

    @property
    def agent_interfaces(self) -> Dict[str, AgentInterface]:
        return self._agent_interfaces

    @property
    def scenario_log(self) -> Dict[str, Union[float, str]]:
        """Simulation steps log.

        Returns:
            (Dict[str, Union[float,str]]): A dictionary with the following keys.
                `fixed_timestep_sec` - Simulation time-step.
                `scenario_map` - Name of the current scenario.
                `scenario_traffic` - Traffic spec(s) used.
                `mission_hash` - Hash identifier for the current scenario.
        """

        scenario = self._smarts.scenario
        return {
            "fixed_timestep_sec": self._smarts.fixed_timestep_sec,
            "scenario_map": scenario.name,
            "scenario_traffic": ",".join(map(os.path.basename, scenario.traffic_specs)),
            "mission_hash": str(hash(frozenset(scenario.missions.items()))),
        }

    @property
    def smarts(self):
        return self._smarts

    @property
    def seed(self):
        return current_seed()
    
    def customizedReset(self, agent_id, options, seed=None):
        super().reset(seed=seed, options=options)
        scenario = next(self._scenarios_iterator)

        self._dones_registered = 0
        observations = self._smarts.reset(
            scenario, start_time=options.get("start_time", 0)
        )
        # printColor(options.get("start_time", 0), 'g')
        # current_vehicles = self._smarts.vehicle_index.social_vehicle_ids(vehicle_types=frozenset({"car"}))
        # printColor(current_vehicles, 'g')
        # vehicle_id = options.get("vehicle_id")

        # mission = Mission.random_endless_mission(scenario._road_map)
        # printColor(vehicle_id)
        # from enviorment.agent_interface import agent_interface
        # self._smarts.add_agent_and_switch_control(vehicle_id, agent_id, agent_interface, mission)

        # sumo_traffic = self._smarts.traffic_sims[0]
        # printColor(sumo_traffic, 'r')
        # printColor(vehicle_id, 'r')
        # sumo_traffic._traci_conn.vehicle.setColor(
        #     vehicle_id, SumoTrafficSimulation._color_for_role(ActorRole.EgoAgent)
        # )

        if self._env_renderer is not None:
            self._env_renderer.reset(observations)

        if seed is not None:
            smarts_seed(seed)
        return self._observations_formatter.format(observations)
    
    # def selected(self, SCENARIO, PERIOD):
    #     data_info = pd.read_csv(ROOT_PATH + "/data/offline_datasets_info/" + SCENARIO + "_candid_list.csv")
    #     self.selected_scenarios = []
    #     for idx,data in data_info.iterrows():
    #         if data["period"] == PERIOD:
    #             id = data["vehicle ID"]
    #             self.selected_scenarios.append({"start_time":, "vehicle_id":})
    #             selected_vehicles.add(f"history-vehicle-{id}")
