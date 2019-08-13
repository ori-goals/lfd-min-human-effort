
import torch
from torch import nn #needed for building neural networks
import torch.nn.functional as F #needed for activation functions
import torch.optim as opt #needed for optimisation
import random
from copy import copy, deepcopy
from collections import deque
import numpy as np

def fanin_(size):
    fan_in = size[0]
    weight = 1./np.sqrt(fan_in)
    return torch.Tensor(size).uniform_(-weight, weight)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, h1=64, h2=32, init_w=3e-3):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(state_dim, h1)
        self.linear1.weight.data = fanin_(self.linear1.weight.data.size())

        #self.bn1 = nn.BatchNorm1d(h1)

        self.linear2 = nn.Linear(h1+action_dim, h2)
        self.linear2.weight.data = fanin_(self.linear2.weight.data.size())

        self.linear3 = nn.Linear(h2, 1)
        self.linear3.weight.data.uniform_(-init_w, init_w)

        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = self.linear1(state)
        x = self.relu(x)
        x = self.linear2(torch.cat([x,action],1))

        x = self.relu(x)
        x = self.linear3(x)

        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, h1=64, h2=32, init_w=0.003):
        super(Actor, self).__init__()

        #self.bn0 = nn.BatchNorm1d(state_dim)

        self.linear1 = nn.Linear(state_dim, h1)
        self.linear1.weight.data = fanin_(self.linear1.weight.data.size())

        #self.bn1 = nn.BatchNorm1d(h1)

        self.linear2 = nn.Linear(h1, h2)
        self.linear2.weight.data = fanin_(self.linear2.weight.data.size())

        #self.bn2 = nn.BatchNorm1d(h2)

        self.linear3 = nn.Linear(h2, action_dim)
        self.linear3.weight.data.uniform_(-init_w, init_w)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        cuda = torch.cuda.is_available() #check for CUDA
        self.device   = torch.device("cuda" if cuda else "cpu")

    def forward(self, state):
        #state = self.bn0(state)
        x = self.linear1(state)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.tanh(x)
        return x

    def get_action(self, state):
        state  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]
