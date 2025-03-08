from typing import Literal
import time
import logging
from dataclasses import dataclass
import vedo

from .air_velocity_scene_actor import SceneParams, Scene, AirVelocitySceneActor


@dataclass(kw_only=True)
class ScenePlayerParams(SceneParams):
    timer_dt_ms: int
    playback_speed: float
    start_scene: int
    start_episode: int
    start_time_s: float


PlayerState = Literal['not_started'] | Literal['running'] | Literal['finished']


class ScenePlayer:

    _log: logging.Logger

    _params: ScenePlayerParams
    _scenes: list[Scene]

    _plotter: vedo.Plotter
    _world: vedo.Box
    _start_stop_button: vedo.Button
    _corner_text: vedo.CornerAnnotation | None

    _timer_callback_id: int | None
    _button_callback_id: int | None
    _timer_id: int | None
    _last_time_s: float | None

    _current_state: PlayerState

    _current_scene_actor: AirVelocitySceneActor | None
    _current_scene_index: int
    _is_show_called: bool

    def __init__(self, params: ScenePlayerParams, scenes: list[Scene]):

        self._log = logging.getLogger(__class__.__name__)

        self._params = params
        self._scenes = scenes

        self._plotter = self._create_plotter()
        self._initialize_callbacks(self._plotter)

        assert len(scenes) > 0, "minimum one scene is required"

        self._is_show_called = False
        self._current_state = 'not_started'
        self._current_scene_actor = None

        if self._params.start_scene >= len(scenes):
            raise ValueError(
                f'invalid start_scene index {self._params.start_scene}; scene count: {len(scenes)}'
            )

        self._current_scene_index = self._params.start_scene - 1
        self._create_next_scene(start_episode=params.start_episode,
                                start_time_s=params.start_time_s)

    def _initialize_callbacks(self, plotter: vedo.Plotter):
        self._timer_callback_id = plotter.add_callback('timer',
                                                       self._timer_handler,
                                                       enable_picking=False)
        self._button_callback_id = plotter.add_callback(
            'end interaction',
            self._end_interaction_handler,
            enable_picking=False)
        self._timer_id = None
        self._last_time_s = None

    def _end_interaction_handler(self, event):
        # reset the time after an interaction to prevent time jumping
        self._last_time_s = time.time()

    def _destroy_callbacks(self):
        if self._timer_callback_id is not None:
            self._plotter.remove_callback(self._timer_callback_id)
            self._timer_callback_id = None

        if self._button_callback_id is not None:
            self._plotter.remove_callback(self._button_callback_id)
            self._button_callback_id = None

    def _create_plotter(self):

        plotter = vedo.Plotter(axes=1)
        plotter.use_depth_peeling(value=True)
        # create the world
        self._world = self._create_world()

        # create start/stop button
        self._start_stop_button = self._create_start_stop_button(plotter)

        self._corner_text = vedo.CornerAnnotation()

        plotter.add(self._world, self._start_stop_button, self._corner_text)
        return plotter

    def _create_world(self):
        box = self._params.simulation_box
        assert box.limit_earth_xyz_low_m is not None and box.limit_earth_xyz_high_m is not None
        pos = [
            box.limit_earth_xyz_low_m[0], box.limit_earth_xyz_high_m[0],
            box.limit_earth_xyz_low_m[1], box.limit_earth_xyz_high_m[1],
            box.limit_earth_xyz_low_m[2], box.limit_earth_xyz_high_m[2]
        ]
        return vedo.Box(pos=pos).wireframe()

    def _create_start_stop_button(self, plotter: vedo.Plotter):

        return plotter.add_button(self._start_stop_button_handler,
                                  states=["\u23F5 Play  ", "\u23F8 Pause"],
                                  font='Kanopus',
                                  size=32)

    def _start_stop_button_handler(self, object, ename):

        play = 'Play' in self._start_stop_button.status()

        if self._timer_id is not None:
            self._plotter.timer_callback('destroy', self._timer_id)
            self._timer_id = None

        if play:
            self._last_time_s = time.time()
            self._timer_id = self._plotter.timer_callback(
                'create', dt=self._params.timer_dt_ms)

        self._start_stop_button.switch()

    def show(self):

        if self._current_state == 'finished':
            return

        axes_opts = dict(text_scale=0.3,
                         xtick_length=0.025,
                         xtick_thickness=0.0015,
                         ytick_thickness=0.0015,
                         xtitle='x (m)',
                         ytitle='y (m)',
                         ztitle='z (m)')

        self._is_show_called = True
        self._plotter.show(viewup='z', axes=axes_opts)

    def _timer_handler(self, _):

        if self._current_state == 'finished':
            return

        if self._current_state == 'not_started':
            self._current_state = 'running'

        if self._last_time_s is None:
            return

        current_time_s = time.time()

        dt_s = current_time_s - self._last_time_s

        self._last_time_s = current_time_s

        assert self._current_scene_actor is not None

        playback_speed = self._params.playback_speed

        scene_status, scene_corner_text = self._current_scene_actor.step(
            dt_s=dt_s * playback_speed, plotter=self._plotter)

        assert self._corner_text is not None
        if len(scene_corner_text) > 0:
            scene_corner_text = '; ' + scene_corner_text
        self._corner_text.text(
            f'speed: {playback_speed}x; scene: {self._current_scene_index}{scene_corner_text}'
        )

        if scene_status == 'finished':
            self._create_next_scene(start_episode=0, start_time_s=0)

        self._plotter.render()

    def _create_next_scene(self, start_episode: int, start_time_s: float):

        self._destroy_current_scene()
        self._current_scene_index += 1

        finished = True
        while len(self._scenes) > self._current_scene_index:
            self._current_scene_actor, scene_state = self._create_scene(
                self._scenes[self._current_scene_index],
                start_episode=start_episode,
                start_time_s=start_time_s)

            if scene_state != 'finished':
                # found a non finished scene
                finished = False
                break

            # try the next
            self._destroy_current_scene()
            self._current_scene_index += 1

        if finished:
            self._current_state = 'finished'
            self._destroy_callbacks()

            if self._is_show_called:
                self._plotter.close()

    def _create_scene(self, scene: Scene, start_episode: int,
                      start_time_s: float):

        scene_actor = AirVelocitySceneActor(scene=scene,
                                            params=self._params,
                                            start_episode=start_episode,
                                            start_time_s=start_time_s)
        scene_state = scene_actor.create(self._plotter)
        return scene_actor, scene_state

    def _destroy_current_scene(self):
        if self._current_scene_actor is None:
            return

        self._current_scene_actor.destroy(self._plotter)
        self._current_scene_actor = None
