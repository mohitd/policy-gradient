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

POLICY_FACTOR = 1
ENTROPY_FACTOR = 0.001
LOG_FREQ = 1_000
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

def _select_action(policy, obs, saved_log_probs, saved_values=list()):
    """Select action
    
    Parameters
    ----------
    policy : nn.Module
        policy network
    obs : ndarray
        observation
    saved_log_probs : list
        log probability for action sampled from the policy
    saved_values : list
        value for the given state
    
    Returns
    -------
    int
        action to take according to our policy
    """
    obs = Variable(torch.from_numpy(obs).type(dtype).permute(2, 0, 1).unsqueeze(0))
    output, value = policy(obs)
    saved_values.append(value)
    action_probs = Categorical(output)
    action = action_probs.sample()
    # record log probability of action and values
    saved_log_probs.append(action_probs.log_prob(action))
    return action.item()

def ppo(env,
        policy_network,
        optimizer_spec,
        num_episodes,
        gamma,
        num_epochs,
        num_steps,
        eps):
    """Runs PPO training algorithm
    
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
    gamma : float
        discount factor
    num_epochs : int
        number of epochs to train network for per episode
    num_steps : int
        max number of steps per episode
    eps : float
        clipping parameter
    """

    # get input sizes and num actions
    img_h, img_w, img_c = env.observation_space.shape
    num_actions = env.action_space.n

    # construct policy network
    policy = policy_network(in_channels=img_c, num_actions=num_actions)
    policy_old = policy_network(in_channels=img_c, num_actions=num_actions)

    # construct optimizer
    optimizer = optimizer_spec.constructor(policy.parameters(), **optimizer_spec.kwargs)

    running_reward = None
    # main training loop
    for episode in range(num_episodes):
        # reset cache
        saved_rewards = []
        saved_old_log_probs = []
        saved_states = []

        # start the environment
        obs = env.reset()
        for t in range(num_steps):
            # select action
            action = _select_action(policy_old, obs, saved_old_log_probs)
            obs, reward, done, _ = env.step(action)
             
            saved_rewards.append(reward)
            saved_states.append(obs)
            if done:
                break

        # episode is finished
        target_state_values = _compute_discounted_rewards(saved_rewards, gamma)

        # optimize policy for K epochs
        for _ in range(num_epochs):
            saved_log_probs = []
            saved_values = []

            for state in saved_states:
                # evaluate new policy on old state in saved_log_probs
                _select_action(policy, state, saved_log_probs, saved_values)
            
            log_probs = torch.cat(saved_log_probs)
            old_log_probs = torch.cat(saved_old_log_probs)

            # compute ratios
            ratios = torch.exp(log_probs - old_log_probs)
            values = Variable(torch.tensor(saved_values))

            # compute clipped policy loss
            advantages = target_state_values - values
            trpo_surrogates = ratios * advantages
            clipped_surrogates = torch.clamp(ratios, 1 - eps, 1 + eps) * advantages
            policy_loss = -torch.min(trpo_surrogates, clipped_surrogates).mean()
            
            value_loss = F.smooth_l1_loss(values, target_state_values).mean()

            entropy = (log_probs.exp() * log_probs).mean()

            loss = value_loss + policy_loss + ENTROPY_FACTOR * entropy

            # update parameters
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # copy over policy
        policy_old.load_state_dict(policy.state_dict())

        # compute running reward
        reward_sum = sum(saved_rewards)
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + episode * 0.01

        # print stats
        print('-' * 64)
        print('Episode {}'.format(episode + 1))
        print('Running reward: {}'.format(running_reward))
        print('Loss: {}'.format(loss))
        print('\n')
        sys.stdout.flush()
