import functools
import os

import embodied
import numpy as np


class DDMC(embodied.Env):

  DEFAULT_CAMERAS = dict(
      locom_rodent=1,
      quadruped=2,
  )

  def __init__(self, env, repeat=1, render=True, size=(64, 64), camera=-1):
    # This env variable is meant for headless GPU machines but may fail on CPU-only machines.
    if 'MUJOCO_GL' not in os.environ:
      os.environ['MUJOCO_GL'] = 'egl'
    if isinstance(env, str):
      domain, task = env.split('_', 1)
      if camera == -1:
        camera = self.DEFAULT_CAMERAS.get(domain, 0)
      if domain == 'cup':  # Only domain with multiple words.
        domain = 'ball_in_cup'

      from adaptgym.envs.distracting_control import suite

      print('Default DDMC config.')
      dynamic = False
      num_videos = 3
      randomize_background = 0
      shuffle_background = 0
      do_color_change = 0
      ground_plane_alpha = 0.1
      background_dataset_videos = ['boat', 'bmx-bumps', 'flamingo']
      continuous_video_frames = True
      do_just_background = True
      difficulty = 'easy'
      specify_background = '0,0,1e6;1,1e6,2e6;0,2e6,1e9'  # ABA

      self._dmenv = suite.load(domain, task, difficulty=difficulty,
                             pixels_only=False, do_just_background=do_just_background,
                             do_color_change=do_color_change,
                             background_dataset_videos=background_dataset_videos,
                             background_kwargs=dict(num_videos=num_videos,
                                                    dynamic=dynamic,
                                                    randomize_background=randomize_background,
                                                    shuffle_buffer_size=shuffle_background * 500,
                                                    seed=1,
                                                    ground_plane_alpha=ground_plane_alpha,
                                                    continuous_video_frames=continuous_video_frames,
                                                    specify_background=specify_background,
                                                    divide_step_count_by=repeat,
                                                    ))

    from . import from_dm
    self._env = from_dm.FromDM(self._dmenv)
    self._env = embodied.wrappers.ExpandScalars(self._env)
    self._env = embodied.wrappers.ActionRepeat(self._env, repeat)
    self._render = render
    self._size = size
    self._camera = camera

  @functools.cached_property
  def obs_space(self):
    spaces = self._env.obs_space.copy()
    if self._render:
      spaces['image'] = embodied.Space(np.uint8, self._size + (3,))
    return spaces

  @functools.cached_property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    for key, space in self.act_space.items():
      if not space.discrete:
        assert np.isfinite(action[key]).all(), (key, action[key])
    obs = self._env.step(action)
    if self._render:
      obs['image'] = self.render()
    return obs

  def render(self):
    return self._dmenv.physics.render(*self._size, camera_id=self._camera)
