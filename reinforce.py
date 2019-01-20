"""
Training algorithm
"""

import sys
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable

EPS = np.finfo(np.float32).eps.item()

# if CUDA, use it
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def _compute_discounted_rewards(rewards, gamma):
    """Compute discounted rewards into the past
    
    Parameters
    ----------
    rewards : list
        reward for each transition
    gamma : float
        discount factor
    
    Returns
    -------
    Variable
        discounted and normalized rewards 
    """

    R = 0
    discounted_rewards = []
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.tensor(discounted_rewards)
    
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + EPS)
    return Variable(discounted_rewards)

def _select_action(policy, obs, saved_log_probs):
    """Select action
    
    Parameters
    ----------
    policy : nn.Module
        policy network
    obs : ndarray
        observation
    saved_log_probs : list
        log probability for action sampled from the policy
    
    Returns
    -------
    int
        action to take according to our policy
    """

    obs = Variable(torch.from_numpy(obs).type(dtype).permute(2, 0, 1).unsqueeze(0))
    output = policy(obs)
    action_probs = Categorical(output)
    action = action_probs.sample()
    # record log probability of action
    saved_log_probs.append(action_probs.log_prob(action))
    return action.item()

def reinforce(env,
        policy_network,
        optimizer_spec,
        num_episodes,
        num_steps,
        gamma,
        grad_norm_clipping):
    """Run REINFORCE algorithm
    
    Parameters
    ----------
    env : gym.Env
        OpenAI gym environment
    policy_network : torch.nn.Module
        policy network that computes a probability distribution over actions
    optimizer_spec : OptimizerSpec
        parameters for the optimizer
    num_episodes : int
        when to stop training: (env, num_timesteps) -> bool
    num_steps : int
        max number of steps per episode
    gamma : float
        discount factor
    grad_norm_clipping : float
        value to clip gradient to
    """

    # get input sizes and num actions
    img_h, img_w, img_c = env.observation_space.shape
    num_actions = env.action_space.n

    # construct policy network
    policy = policy_network(in_channels=img_c, num_actions=num_actions)

    # construct optimizer
    optimizer = optimizer_spec.constructor(policy.parameters(), **optimizer_spec.kwargs)

    running_reward = None
    # main training loop
    for episode in range(num_episodes):
        # reset cache
        saved_rewards = []
        saved_log_probs = []

        # start the environment
        obs = env.reset()

        for t in range(num_steps):
            # select action
            action = _select_action(policy, obs, saved_log_probs)
            obs, reward, done, _ = env.step(action)
            # record reward
            saved_rewards.append(reward)
            if done:
                break
        
        # episode is finished so compute episode loss
        discounted_rewards = _compute_discounted_rewards(saved_rewards, gamma)
        saved_log_probs = torch.cat(saved_log_probs)
        policy_loss = (-saved_log_probs * discounted_rewards).sum()

        # update parameters
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # compute running reward
        reward_sum = sum(saved_rewards)
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + episode * 0.01

        # print stats
        print('-' * 64)
        print('Episode {}'.format(episode + 1))
        print('Running reward: {}'.format(running_reward))
        print('Total loss: {}'.format(policy_loss))
        print('\n')
        sys.stdout.flush()

    # end main training loop
# end REINFORCE function
