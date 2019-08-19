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
        self.q_optimizer  = opt.Adam(self.critic.parameters(),  lr=config.lr_critic, weight_decay=0.01)
        self.policy_optimizer = opt.Adam(self.actor.parameters(), lr=config.lr_actor, weight_decay=0.01)
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        self.lr_actor = config.lr_actor
        self.lr_bc = config.lr_bc
        self.min_buffer_size = config.min_buffer_size
        self.rl_batch_size = config.rl_batch_size
        self.demo_batch_size = config.demo_batch_size
        self.q_filter_epsilon = config.q_filter_epsilon
        self.tau = config.tau
        self.gamma = config.gamma
        self.noise_factor = config.noise_factor
        self.plot_reward = []
        self.plot_average_rewards = []
        self.plot_policy = []
        self.plot_q = []
        self.plot_steps = []
        self.noise_decay = config.noise_decay
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

    def get_batch(self, rl_replay_buffer, demo_replay_buffer):
        '''
        If each of the replay buffers exceeds the minimum sample size, both
        the rl and demo replay buffers are sampled. The full_batch returns
        the entire sample, which might include samples from both the rl and
        demo buffer, and demo_batch is the sample from the demo buffer only.
        '''
        if (rl_replay_buffer is not None) and (demo_replay_buffer is None):
            full_batch = rl_replay_buffer.sample(self.rl_batch_size)
            demo_batch = None
        elif (demo_replay_buffer is not None) and (rl_replay_buffer is None):
            demo_batch = demo_replay_buffer.sample(self.demo_batch_size)
            full_batch = demo_batch
        elif (demo_replay_buffer is not None) and (rl_replay_buffer is not None):
            demo_batch = demo_replay_buffer.sample(self.demo_batch_size)
            rl_batch = rl_replay_buffer.sample(self.rl_batch_size)
            full_batch = []
            for index in range(5):
                sample = np.concatenate((demo_batch[index], rl_batch[index]))
                full_batch.append(sample)
            full_batch = tuple(full_batch)
        return full_batch, demo_batch

    def update(self, rl_replay_buffer=None, demo_replay_buffer=None):
        '''
        If the replay buffers are large enough, performs the update from Nair et
        al Overcoming Exploration in Reinforcement Learning
        with Demonstrations. This consists of performing a normal DDPG
        update on experience sampled both from demonstrations and RL.

        Additionally, a behaviour cloning update is performed on the demo batch
        if each sample passes the "Q-filter".
        '''

        full_batch, demo_batch = self.get_batch(rl_replay_buffer, demo_replay_buffer)
        s_batch, a_batch, r_batch, t_batch, s2_batch = full_batch
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

        # critic update
        self.q_optimizer.zero_grad()
        MSE = nn.MSELoss()
        q_loss = MSE(q, y)
        q_loss.backward()
        self.q_optimizer.step()

        # compute loss for actor
        self.policy_optimizer.zero_grad()
        policy_loss = -self.critic(s_batch, self.actor(s_batch))

        # if demonstrations compute the behaviour cloning loss
        if demo_batch is not None:
            dem_s_batch, dem_a_batch, dem_r_batch, dem_t_batch, dem_s2_batch = demo_batch
            dem_s_batch = torch.FloatTensor(dem_s_batch).to(self.device)
            dem_a_batch = torch.FloatTensor(dem_a_batch).to(self.device)
            pi_a = self.actor(dem_s_batch)

            dem_qval = self.critic(dem_s_batch, dem_a_batch).detach()
            pi_qval = self.critic(dem_s_batch, pi_a).detach()
            mask = dem_qval > pi_qval - pi_qval.abs()*self.q_filter_epsilon # don't apply q-filter update if policy action deemed better
            bc_loss = MSE(pi_a*mask.type(torch.FloatTensor), dem_a_batch*mask.type(torch.FloatTensor))  # multiplying by the mask means if the mask is zero (policy action is better)
                                                        # that state does not contribute to the behaviour cloning loss
            policy_loss += self.lr_bc/self.lr_actor*bc_loss #scale the bc loss according to the learning rate
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
