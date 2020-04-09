from typing import Optional, Type

import habitat
from habitat import Config, Dataset
from habitat_baselines.common.baseline_registry import baseline_registry
from utils.vis_utils import observations_to_image, append_text_to_image
import cv2
def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="CustomObjectNavRLEnv")
class CustomObjectNavEnv(habitat.RLEnv):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE

        self._previous_measure = None
        self._previous_action = None
        self.time_t = 0
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        self.time_t = 0
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]
        self.obs = observations
        self.info = None
        self.total_reward = 0
        return observations

    def step(self, action):
        self._previous_action = action
        obs, reward, done, self.info = super().step(action)
        self.time_t += 1
        self.info['length'] = self.time_t * done
        self.obs = obs
        self.total_reward += reward
        return obs, reward, done, self.info

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    def render(self, mode='rgb'):
        info = self.get_info(None) if self.info is None else self.info
        img = observations_to_image(self.obs, info)
        str_action = ''
        if self._previous_action is not None:
            action_list = ["STOP", "MOVE_FORWARD", 'TURN_LEFT', 'TURN_RIGHT']
            str_action = action_list[self._previous_action['action']]
        img = append_text_to_image(img, 't: %03d, r: %f a: %s'%(self.time_t,self.total_reward, str_action))
        if mode == 'rgb' or mode == 'rgb_array':
            return img
        elif mode == 'human':
            cv2.imshow('render', img)
            cv2.waitKey(1)
            return img
        return super().render(mode)
