import threading
from typing import Any, cast, Literal
import queue
from threading import Thread
import time
from env.glider.base.agent.config import register_glider_env_agent_config_groups
from env.glider.base.visualization.interactive_trajectory_visualization import InteractiveTrajectoryVisualization
from env.glider.mcts.glider_mcts_adapter import GliderEnvAdapterParams, GliderEnvLogger
import hydra
from hydra.core import hydra_config
from functools import partial
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING
from gymnasium.utils import seeding

from env.glider.base import (AgentID, GliderSimulationParameters,
                             GliderTrajectory,
                             register_glider_env_aerodynamics_config_groups)
from env.glider.base.agent import (GliderAgent,
                                   GliderInitialConditionsCalculator,
                                   GliderInitialConditionsParameters,
                                   GliderRewardAndCutoffCalculator)
from thermal.gaussian import register_gaussian_air_velocity_field_config_groups

from env.glider.mcts import GliderAgentState, GliderEnvAdapter, DiscreteToContinuousDegreesParams
from mcts.mcts_worker import MCTSParameters, MCTSWorker


@dataclass
class SimulationParameters(GliderSimulationParameters):
    discrete_continuous_mapping: DiscreteToContinuousDegreesParams
    initial_conditions_params: GliderInitialConditionsParameters = field(
        default_factory=GliderInitialConditionsParameters)


@dataclass
class ExperimentConfig:
    experiment_name: str | None
    seed: int | None
    render: bool
    simulation: SimulationParameters
    mcts: MCTSParameters
    glider_env_adapter: GliderEnvAdapterParams

    air_velocity_field: Any = MISSING
    aerodynamics: Any = MISSING


def initialize_config_store():

    # register config instances
    cs = ConfigStore.instance()
    cs.store(name='experiment_config', node=ExperimentConfig)
    cs.store(name='simulation', node=SimulationParameters)

    register_gaussian_air_velocity_field_config_groups(
        group='env/glider/air_velocity_field', config_store=cs)
    register_glider_env_aerodynamics_config_groups(config_store=cs)
    register_glider_env_agent_config_groups(config_store=cs)


def create_simulation_environment(cfg: ExperimentConfig):

    aerodynamics = hydra.utils.instantiate(cfg.aerodynamics,
                                           _convert_='object')
    air_velocity_field = hydra.utils.instantiate(cfg.air_velocity_field,
                                                 _convert_='object')
    simulation_params = cfg.simulation

    initial_conditions_calculator = GliderInitialConditionsCalculator(
        initial_conditions_params=simulation_params.initial_conditions_params,
        air_velocity_field=air_velocity_field,
        aerodynamics=aerodynamics)

    reward_and_cutoff_calculator = GliderRewardAndCutoffCalculator(
        cutoff_params=simulation_params.cutoff_params,
        reward_params=simulation_params.reward_params,
        simulation_box=simulation_params.simulation_box_params)

    seed = cfg.seed
    np_random, _ = seeding.np_random(seed)
    initial_conditions_calculator.seed(seed)
    air_velocity_field.seed(seed)

    agent_params = hydra.utils.instantiate(
        simulation_params.glider_agent_params, _convert_='object')

    agent = GliderAgent(
        agent_id=AgentID('mcts'),
        parameters=agent_params,
        initial_conditions_calculator=initial_conditions_calculator,
        reward_and_cutoff_calculator=reward_and_cutoff_calculator,
        aerodynamics=aerodynamics,
        air_velocity_field=air_velocity_field,
        np_random=np_random)

    air_initial_conditions_info: dict = air_velocity_field.reset()

    agent.reset(time_s=0.)

    agent_state = GliderAgentState(
        agent,
        air_initial_conditions_info=air_initial_conditions_info,
        time_s=0.,
        cumulative_reward=0.)

    adapter = create_mcts_adapter(cfg)
    logger = GliderEnvLogger()

    adapter.seed(cfg.seed)

    return agent_state, adapter, logger


def create_mcts_adapter(config: ExperimentConfig):

    simulation_params = config.simulation

    output_dir = hydra_config.HydraConfig.get().runtime.output_dir
    adapter = GliderEnvAdapter(params=config.glider_env_adapter,
                               discrete_to_continuous_params=simulation_params.
                               discrete_continuous_mapping,
                               time_params=simulation_params.time_params,
                               snapshot_out_dir=output_dir)

    return adapter


def do_the_job(state: GliderAgentState, worker: MCTSWorker):
    worker.start(state=state, stop_event=None, render_queue=None)


def do_the_job_with_rendering(state: GliderAgentState, worker: MCTSWorker):

    visual = InteractiveTrajectoryVisualization()

    job_queue = queue.Queue[GliderTrajectory | Literal['done']]()
    job_exception = queue.Queue[Exception]()

    # used to signal the worker thread to stop
    stop_event = threading.Event()

    def worker_func():

        try:
            worker.start(state=state,
                         stop_event=stop_event,
                         render_queue=job_queue)

            job_queue.put('done')
        except Exception as e:
            job_exception.put(e)
            job_queue.put('done')

    # create the worker thread
    worker_thread = Thread(target=worker_func)
    worker_thread.start()

    visual.show()

    # start rendering on the main thread
    try:
        while True:

            if not job_queue.empty():
                data = job_queue.get_nowait()

                if data == 'done':
                    break

                if data is not None:
                    visual.update(data)

            visual.render()
            time.sleep(0.01)

    except (Exception, KeyboardInterrupt) as e:
        # stop and wait the worker thread
        stop_event.set()
        worker_thread.join()

        raise e
    finally:
        visual.close()

    if not job_exception.empty():
        raise job_exception.get()


initialize_config_store()


@hydra.main(version_base=None,
            config_name='thermal_baseline_mcts_config',
            config_path='config')
def exp_main(cfg: ExperimentConfig):

    experiment_config = cast(ExperimentConfig, OmegaConf.to_object(cfg))

    agent_state, adapter, logger = create_simulation_environment(
        experiment_config)

    worker = MCTSWorker(env_adapter=adapter,
                        env_logger=logger,
                        params=experiment_config.mcts)
    worker.seed(experiment_config.seed)

    if experiment_config.render:
        do_the_job_with_rendering(state=agent_state, worker=worker)
    else:
        do_the_job(state=agent_state, worker=worker)


if __name__ == '__main__':
    exp_main()
