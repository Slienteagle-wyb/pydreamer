import time

import torch
from distutils.util import strtobool
from pydreamer.models import dreamer
from pydreamer.envs import create_env
from generator import NetworkPolicy
from pydreamer.preprocessing import Preprocessor
import argparse
from pydreamer import tools

if __name__ == '__main__':
    model_path = f'/home/yibo/Documents/pydreamer/mlruns/0/epoch=163500.pt'
    check_points = torch.load(model_path, map_location=torch.device('cpu'))
    model_state_dict = check_points['model_state_dict']

    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()
    # Config from YAML
    conf = {}
    configs = tools.read_yamls('./config')
    for name in args.configs:
        if ',' in name:
            for n in name.split(','):
                conf.update(configs[n])
        else:
            conf.update(configs[name])
    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = lambda x: bool(strtobool(x))
        parser.add_argument(f'--{key}', type=type_, default=value)
    conf = parser.parse_args(remaining)

    model = dreamer.Dreamer(conf)
    model.load_state_dict(model_state_dict)

    env = create_env(conf.env_id, conf.env_no_terminal, conf.env_time_limit, conf.env_action_repeat)
    preprocess = Preprocessor(image_categorical=conf.image_channels if conf.image_categorical else None,
                              image_key=conf.image_key,
                              map_categorical=conf.map_channels if conf.map_categorical else None,
                              map_key=conf.map_key,
                              action_dim=env.action_size,  # type: ignore
                              clip_rewards=conf.clip_rewards)
    policy = NetworkPolicy(model, preprocess)

    obs = env.reset()
    done = False

    for i in range(8000):
        action, metric = policy(obs)
        obs, reward, done, inf = env.step(action)
        if done:
            print(reward)
            done = False
            obs = env.reset()
