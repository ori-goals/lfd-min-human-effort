import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from ddpg_models import *
from utils import *

class DDPGAgent(object):
    def __init__(self, config, state_dim, action_dim):
        cuda = torch.cuda.is_available() #check for CUDA
        self.device   = torch.device("cuda" if cuda else "cpu")
        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
        self.critic  = Critic(state_dim, action_dim).to(self.device)
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.target_critic  = Critic(state_dim, action_dim).to(self.device)
        self.target_actor = Actor(state_dim, action_dim).to(self.device)
        self.q_optimizer  = opt.Adam(self.critic.parameters(),  lr=config.lr_critic)#, weight_decay=0.01)
        self.policy_optimizer = opt.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        self.plot_reward = []
        self.plot_average_rewards = []
        self.plot_policy = []
        self.plot_q = []
        self.plot_steps = []
        self.min_buffer_size = config.min_buffer_size
        self.rl_batch_size = config.rl_batch_size
        self.demo_batch_size = config.demo_batch_size
        self.tau = config.tau
        self.gamma = config.gamma
        self.noise_factor = config.noise_factor
        self.copy_networks()

    def copy_networks(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

    def add_plotting_data(self, episode_reward, step, episode_number):
        self.plot_reward.append([episode_reward, episode_number+1])
        self.plot_steps.append([step+1, episode_number+1])
        window = 10
        sum = 0.0
        if len(self.plot_reward) > window:
            for entry in self.plot_reward[-window:]:
                sum += entry[0]
            self.plot_average_rewards.append([sum/window, episode_number+1])
        try:
            self.plot_policy.append([self.policy_loss.data, episode_number+1])
            self.plot_q.append([self.q_loss.data, episode_number+1])
        except:
            pass

    def update(self, rl_replay_buffer=None, demo_replay_buffer=None):
        s_batch, a_batch, r_batch, t_batch, s2_batch = rl_replay_buffer.sample(self.rl_batch_size)
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        a_batch = torch.FloatTensor(a_batch).to(self.device)
        r_batch = torch.FloatTensor(r_batch).unsqueeze(1).to(self.device)
        t_batch = torch.FloatTensor(np.float32(t_batch)).unsqueeze(1).to(self.device)
        s2_batch = torch.FloatTensor(s2_batch).to(self.device)


        #compute loss for critic
        a2_batch = self.target_actor(s2_batch)
        target_q = self.target_critic(s2_batch, a2_batch) #detach to avoid updating target
        y = r_batch + (1.0 - t_batch) * self.gamma * target_q.detach()
        q = self.critic(s_batch, a_batch)

        self.q_optimizer.zero_grad()
        MSE = nn.MSELoss()
        q_loss = MSE(q, y) #detach to avoid updating target
        q_loss.backward()
        self.q_optimizer.step()

        #compute loss for actor
        self.policy_optimizer.zero_grad()
        policy_loss = -self.critic(s_batch, self.actor(s_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.policy_optimizer.step()

        #soft update of the frozen target networks
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        self.policy_loss = policy_loss
        self.q_loss = q_loss
