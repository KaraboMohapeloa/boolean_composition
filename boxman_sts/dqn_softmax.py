import numpy as np
import random
import gym
import os
import deepdish as dd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def __call__(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class ReplayBuffer(object):
    def __init__(self, size, N=-100):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.N = N
        self.goals = []
        self.goals_hash = []
        if os.path.exists('./goals.h5'):
            self.goals = dd.io.load('./goals.h5')
            for goal in self.goals:
                self.goals_hash.append(goal.sum())

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)
        """
        if done:     
            obs_t_hash = obs_t.sum()
            if obs_t_hash not in self.goals_hash:
                self.goals.append(obs_t)
                self.goals_hash.append(obs_t_hash)
                dd.io.save('goals.h5', self.goals, compression=None)   
                print("\nGoals saved: ",len(self.goals),"\n")  
        """    
            
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize 

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        obses_goal_t, actions, rewards, obses_goal_tp1, dones = [], [], [], [], []
        lg = len(self.goals)
        ng = np.arange(lg)
        np.random.shuffle(ng)  
        mbs = int(batch_size/lg)
        indices = np.random.randint(0, len(self._storage), mbs)             
                
        for i in range(batch_size):
            obs_t, action, reward, obs_tp1, done = self._storage[indices[i%mbs]] 
            obs_t = np.array(obs_t, copy=False)
            obs_tp1 = np.array(obs_tp1, copy=False)               
            
            goal = self.goals[ng[int(i/mbs)%lg]]
            if done and obs_t.sum() != goal.sum() :
                reward = -2 #self.N   
            
            obses_goal_t.append(np.concatenate((obs_t,goal),axis=2))
            actions.append(np.array(action.cpu(), copy=False))
            rewards.append(reward)
            obses_goal_tp1.append(np.concatenate((obs_tp1,goal),axis=2))
            dones.append(done)
        return np.array(obses_goal_t), np.array(actions), np.array(rewards), np.array(obses_goal_tp1), np.array(dones)


class DQN(nn.Module):
    def __init__(self, n_action):
        super(DQN, self).__init__()
        self.n_action = n_action

        # for grey scale
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        # for single frame colour
        self.conv1 = nn.Conv2d(3+3, 32, kernel_size=8, stride=4)
        # for 2 stacked frames colour
        # self.conv1 = nn.Conv2d(6, 32, kernel_size=8, stride=4)
        # for 3 stacked frames colour
        # self.conv1 = nn.Conv2d(9, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.linear1 = nn.Linear(3136, 512)
        # self.linear1 = nn.Linear(18496, 512)
        self.head = nn.Linear(512, self.n_action)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = F.relu(self.linear1(x.reshape(x.size(0), -1)))
        x = self.head(x)
        return x.squeeze()

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        # copy params
        for name, param in state_dict.items():
            own_state[name].copy_(param)
        # freeze params
        for name, param in self.named_parameters():
            if name.split(".")[0] in ["conv1","conv2","conv3"]:
                param.requires_grad = False


class ComposedDQN(nn.Module):
    def __init__(self, dqns, compose="or", rmax=2, rmin=-0.1):
        super(ComposedDQN, self).__init__()
        self.compose = compose
        self.dqns = dqns
        self.rmax = rmax
        self.rmin = rmin
        self.dqn_max = MaxDQN(dqns[0], self.rmax)
    
    def forward(self, obs_goal):
        qs = [self.dqns[i](obs_goal) for i in range(len(self.dqns))]
        qs = torch.stack(tuple(qs), 0)
        if self.compose=="or":
            q = qs.max(0)[0]
        elif self.compose=="and":
            q = qs.min(0)[0]
        else: #not
            q_max = self.dqn_max(obs_goal)
            q_min = q_max - (self.rmax-self.rmin)
            q = (q_max+q_min)-qs[0]

        return q.detach().clone()


class MaxDQN(nn.Module):
    def __init__(self, dqn, rmax=2):
        super(MaxDQN, self).__init__()
        self.dqn = dqn
        self.rmax = rmax
    
    def forward(self, obs_goal):
        dqn_max = self.dqn(obs_goal)
        s = obs_goal[:,:,:,:3]
        g = obs_goal[:,:,:,3:]        
        if s.sum() != g.sum():
            q_gg = self.dqn(torch.cat((g,g),dim=3))
            c = self.rmax-q_gg.max()
            dqn_max = dqn_max + c
        else:
            dqn_max = dqn_max*0 + self.rmax        
        return dqn_max

class Agent(object):
    def __init__(self,
                 env,
                 training_stats = {"R": [], "T": 0},
                 max_timesteps=2000000,
                 learning_starts=10000,
                 train_freq=4,
                 target_update_freq=1000,
                 learning_rate=1e-4,
                 batch_size=128,
                 replay_buffer_size=300000,
                 gamma=0.99,
                 eps_initial=1.0,
                 eps_final=0.01,
                 eps_timesteps=1000000,
                 print_freq=10,
                 path=None):
        assert type(env.observation_space) == gym.spaces.Box
        assert type(env.action_space) == gym.spaces.Discrete

        self.env = env
        self.max_timesteps = max_timesteps
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.gamma = gamma
        self.print_freq = print_freq
        self.path = path
        self.training_stats = {"R": [], "T": 0}

        self.eps_schedule = LinearSchedule(eps_timesteps, eps_final, eps_initial)

        self.q_func = DQN(self.env.action_space.n)
        self.target_q_func = DQN(self.env.action_space.n)
        self.target_q_func.load_state_dict(self.q_func.state_dict())

        if use_cuda:
            self.q_func.cuda()
            self.target_q_func.cuda()

        self.optimizer = optim.Adam(self.q_func.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(replay_buffer_size, N=(env.rmin-env.rmax)*env.diameter)
        self.steps = 0

    def select_action(self, obs, tau_max=5, tau_min=0.01):
        """
        Boltzmann (softmax) exploration with annealed tau based on eps_threshold.
        tau decreases from tau_max to tau_min as eps_threshold decreases.
        """
        eps_threshold = self.eps_schedule(self.steps)
        # Anneal tau: tau = tau_min + (tau_max - tau_min) * eps_threshold
        tau = tau_min + (tau_max - tau_min) * eps_threshold
        if len(self.replay_buffer.goals) > 0:
            obs = np.array(obs)
            obs = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0)
            values = []
            for goal in self.replay_buffer.goals:
                goal = torch.from_numpy(np.array(goal)).type(FloatTensor).unsqueeze(0)
                x = torch.cat((obs, goal), dim=3)
                q_vals = self.q_func(x).squeeze(0)  # shape: [n_actions]
                values.append(q_vals)
            # values: [n_goals, n_actions] -> take max over goals for each action
            values = torch.stack(values, 0)  # shape: [n_goals, n_actions]
            q_per_action = values.max(0)[0]  # shape: [n_actions]
            # Softmax over actions
            probs = torch.softmax(q_per_action / tau, dim=0)
            action = torch.multinomial(probs, 1).reshape(1, 1)
            return action
        else:
            sample_action = self.env.action_space.sample()
            return torch.IntTensor([[sample_action]])

    def train(self):
        obs = self.env.reset()
        self.training_stats = {"R": [], "T": 0}  # Reset stats at start
        k = 0
        T = 0
        self.training_stats["R"].append(0.0)

        for t in range(self.max_timesteps):
            action = self.select_action(obs)
            new_obs, reward, done, info = self.env.step(int(action[0][0]))
            self.replay_buffer.add(obs, action.cpu(), reward, new_obs, done)
            obs = new_obs

            self.training_stats["R"][k] += reward
            T += 1

            if done:
                obs = self.env.reset()
                self.training_stats["R"].append(0.0)
                k += 1

            if t > self.learning_starts and t % self.train_freq == 0:
                obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.replay_buffer.sample(self.batch_size)
                obs_batch = Variable(torch.from_numpy(obs_batch).type(FloatTensor))
                act_batch = Variable(torch.from_numpy(act_batch).type(LongTensor))
                rew_batch = Variable(torch.from_numpy(rew_batch).type(FloatTensor))
                next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(FloatTensor))
                not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(FloatTensor)

                if use_cuda:
                    act_batch = act_batch.cuda()
                    rew_batch = rew_batch.cuda()

                current_q_values = self.q_func(obs_batch).gather(1, act_batch.squeeze(2)).squeeze()
                next_max_q = self.target_q_func(next_obs_batch).detach().max(1)[0]
                next_q_values = not_done_mask * next_max_q
                target_q_values = rew_batch + (self.gamma * next_q_values)

                loss = F.smooth_l1_loss(current_q_values, target_q_values)

                self.optimizer.zero_grad()
                loss.backward()
                for params in self.q_func.parameters():
                    params.grad.data.clamp_(-1, 1)
                self.optimizer.step()

            # Periodically update the target network by Q network to target Q network
            if t > self.learning_starts and t % self.target_update_freq == 0:
                self.target_q_func.load_state_dict(self.q_func.state_dict())
                torch.save(self.q_func.state_dict(), self.path+'model.dqn')
                dd.io.save('exps_data/softmax/prime_experiment_stats.h5', self.training_stats)
                print("\nModel and training stats saved\n")

            self.steps += 1

            mean_100ep_reward = round(np.mean(self.training_stats["R"][-101:-1]), 1)
            num_episodes = len(self.training_stats["R"])
            if done and self.print_freq is not None and num_episodes % self.print_freq == 0:
                print("--------------------------------------------------------")
                print("steps {}".format(t))
                print("episodes {}".format(num_episodes))
                print("mean 100 episode reward {}".format(mean_100ep_reward))
                print("% time spent exploring {}".format(int(100 * self.eps_schedule(t))))
                print("--------------------------------------------------------")

        self.training_stats["T"] = T
                
