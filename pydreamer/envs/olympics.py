import random
import numpy as np
import gym

from rl_trainer.algo.random import random_agent
from olympics_engine.generator import create_scenario
from olympics_engine.scenario import table_hockey, football, wrestling, Running_competition

RENDER = False
ACTION_SET = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18],
              5: [-100, 30], 6: [-40, -30], 7: [-40, -18], 8: [-40, -6], 9: [-40, 6],
              10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18], 14: [20, -6],
              15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18],
              20: [80, -6], 21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30],
              25: [140, -18], 26: [140, -6], 27: [140, 6], 28: [140, 18], 29: [140, 30],
              30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
              35: [200, 30]}   # dicretise action space

ALL_GAMES = frozenset([
    'running-competition',
    'table-hockey',
    'football',
    'wrestling',
    'integrated'
])


def make_env(sub_game, max_steps):
    if sub_game == 'running-competition':
        map_id = random.randint(1, 4)
        game_map = create_scenario(sub_game)
        env = Running_competition(meta_map=game_map, map_id=map_id,
                                  vis=200, vis_clear=5, agent1_color='light red',
                                  agent2_color='blue')
        env.max_step = max_steps
    elif sub_game == 'table-hockey':
        game_map = create_scenario(sub_game)
        env = table_hockey(game_map)
        env.max_step = 400
    elif sub_game == 'football':
        game_map = create_scenario(sub_game)
        env = football(game_map)
        env.max_step = 400
    elif sub_game == 'wrestling':
        game_map = create_scenario(sub_game)
        env = wrestling(game_map)
        env.max_step = 400
    else:
        raise NotImplementedError
    return env


class Olympics(gym.Env):
    def __init__(self, level, num_action_repeat=1, ctrl_agent_idx=1,
                 num_agent=2, max_step=400, action_set=ACTION_SET):
        super().__init__()
        self.sub_game = level
        self.num_action_repeat = num_action_repeat
        self.ctrl_agent_idx = ctrl_agent_idx
        self.num_agent = num_agent
        self.current_energy = 1000

        self.env = make_env(
            self.sub_game,
            max_step
        )

        self.action_set = action_set
        self.action_space = gym.spaces.Discrete(len(action_set))
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(40, 40, 1), dtype=np.uint8)

    def reset(self):
        state = self.env.reset()
        if RENDER:
            self.env.render()
        return self.observation(state)

    def observation(self, state):
        if isinstance(state[self.ctrl_agent_idx], type({})):
            obs_ctrl_agent, energy_ctrl_agent = state[self.ctrl_agent_idx]['agent_obs'], \
                                                self.env.agent_list[self.ctrl_agent_idx].energy
            obs_oppo_agent, energy_oppo_agent = state[1 - self.ctrl_agent_idx]['agent_obs'], \
                                                self.env.agent_list[1 - self.ctrl_agent_idx].energy
        else:
            obs_ctrl_agent, energy_ctrl_agent = state[self.ctrl_agent_idx], \
                                                self.env.agent_list[self.ctrl_agent_idx].energy
            obs_oppo_agent, energy_oppo_agent = state[1 - self.ctrl_agent_idx], \
                                                self.env.agent_list[1 - self.ctrl_agent_idx].energy
        obs_ctrl_agent = np.expand_dims(obs_ctrl_agent, axis=-1).astype(np.uint8)
        self.current_energy = energy_ctrl_agent
        return obs_ctrl_agent

    def step(self, actions):
        action_ctrl = ACTION_SET[actions]
        action_opponent = ACTION_SET[random.randint(0, 35)]
        state, reward, done, _ = self.env.step([action_opponent, action_ctrl])
        obs = self.observation(state)
        reward = self.get_reward(reward)

        if RENDER:
            self.env.render()
        return obs, reward, done, {}

    def get_reward(self, reward_raw):
        reward_finished = 10 * reward_raw[1]
        # reward_curiosity = (1000 - self.current_energy) / 1000.0
        reward = reward_finished
        return reward


if __name__ == '__main__':
    env = Olympics(level='football')
    obs = env.reset()
    ctrl_agent = random_agent()
    oppo_agent = random_agent()
    done = False
    for i in range(12000):
        obs, reward, done, _ = env.step(random.randint(0, 35))
        if done:
            env.reset()
            done = False
            print(reward)

