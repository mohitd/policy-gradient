"""
Policy gradient models
"""

import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):
        """Policy gradient following the architecture of DeepMind's nature paper
        
        Parameters
        ----------
        in_channels : int, optional
            number of input channels, i.e., stacked frames (the default is 4)
        num_actions : int, optional
            number of discrete actions we can take (the default is 18)
        """
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)

        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # flatten
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc4(x))

        # actor head
        actions = self.actor(x)
        action_dist = F.softmax(actions, dim=1)

        # critic head
        value = self.critic(x)

        return action_dist, value
