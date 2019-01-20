"""
Main file to run
"""

import gym
import random
import argparse
from collections import namedtuple

import torch.optim as optim

from utils.gym_envs import get_env

from policy_network import Policy
from actorcritic import ActorCritic

from reinforce import reinforce
from a2c import a2c
from ppo import ppo

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

ENVS = ['PongNoFrameskip-v4', 'BreakoutNoFrameskip-v4']
MODELS = ['vanilla', 'a2c', 'ppo']

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--env', help='Atari environment', choices=ENVS, default=ENVS[0])
parser.add_argument('-f', '--flavor', help='Flavor of model', choices=MODELS, default=MODELS[0])

# optimizer params
LEARNING_RATE = 0.001

# training params
GAMMA = 0.99
UPDATE_FREQ = 20
GRAD_NORM_CLIPPING = 10
NUM_STEPS = 10_000
NUM_EPOCHS = 5
EPS = 0.2

def atari_learn(env, args, num_episodes):
    """Trains policy gradient algorithms on atari environment
    
    Parameters
    ----------
    env : gym.Env
        OpenAI Gymgym environment
    num_episodes : int
        maximum number of episodes
    """

    optimizer = OptimizerSpec(constructor=optim.Adam, kwargs={'lr': LEARNING_RATE})

    if args.flavor == 'vanilla':
        reinforce(env=env,
            policy_network=Policy,
            optimizer_spec=optimizer,
            num_episodes=num_episodes,
            num_steps=NUM_STEPS,
            gamma=GAMMA,
            grad_norm_clipping=GRAD_NORM_CLIPPING)
    elif args.flavor == 'a2c':
        a2c(env=env,
            policy_network=ActorCritic,
            optimizer_spec=optimizer,
            num_episodes=num_episodes,
            gamma=GAMMA,
            update_freq=UPDATE_FREQ,
            grad_norm_clipping=GRAD_NORM_CLIPPING)
    elif args.flavor == 'ppo':
        ppo(env=env,
            policy_network=ActorCritic,
            optimizer_spec=optimizer,
            num_episodes=num_episodes,
            gamma=GAMMA,
            num_epochs=NUM_EPOCHS,
            num_steps=NUM_STEPS,
            eps=EPS)

if __name__ == '__main__':
    args = parser.parse_args()

    # create environment and generate random seed
    task = gym.make(args.env)
    seed = random.randint(0, 9999)
    print('random seed = %d' % seed)

    # wrap environment in the same style as DeepMind
    env = get_env(task, seed)

    atari_learn(env, args, num_episodes=10_000)
