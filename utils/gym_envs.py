"""
    This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import torch
import random
import numpy as np

import gym
from gym import wrappers

from utils.atari_wrappers import wrap_deepmind, wrap_deepmind_ram

def set_global_seeds(i):
    torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)

def get_env(task, seed):
    env = task

    expt_dir = 'tmp/gym-results'
    env = wrappers.Monitor(env, expt_dir, force=True)
    env = wrap_deepmind(env)

    return env

def get_ram_env(env, seed):
    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = 'tmp/gym-results'
    env = wrappers.Monitor(env, expt_dir, force=True)
    env = wrap_deepmind_ram(env)

    return env

def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s"%classname)
