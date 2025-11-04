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
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class BootstrappedReplayBuffer(object):
    def __init__(self, size, num_heads=10, max_goals=1000):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.num_heads = num_heads
        self.max_goals = max_goals
        self.goals = []
        self.goals_hash = []
        
        # Load existing goals if available
        if os.path.exists('./goals.h5'):
            self.goals = dd.io.load('./goals.h5')
            for goal in self.goals:
                self.goals_hash.append(goal.sum())
            # Trim if exceeds max_goals
            if len(self.goals) > self.max_goals:
                self.goals = self.goals[:self.max_goals]
                self.goals_hash = self.goals_hash[:self.max_goals]

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)
        
        # Save terminal states as potential goals
        if done:     
            obs_t_hash = obs_t.sum()
            if obs_t_hash not in self.goals_hash and len(self.goals) < self.max_goals:
                self.goals.append(obs_t.copy())
                self.goals_hash.append(obs_t_hash)
                dd.io.save('goals.h5', self.goals, compression=None)   
                print(f"\nGoals saved: {len(self.goals)}\n")  
            
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size, head_idx=None):
        """Sample batch with goal conditioning and bootstrap masking"""
        if len(self.goals) == 0:
            # Fallback to regular sampling if no goals
            indices = np.random.randint(0, len(self._storage), batch_size)
            obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
            for idx in indices:
                obs_t, action, reward, obs_tp1, done = self._storage[idx]
                obses_t.append(obs_t)
                actions.append(action)
                rewards.append(reward)
                obses_tp1.append(obs_tp1)
                dones.append(done)
            
            # Create bootstrap mask
            if head_idx is not None:
                mask = (torch.rand(batch_size) < 0.5).float()
            else:
                mask = torch.ones(batch_size)
                
            return (np.array(obses_t), np.array(actions), np.array(rewards), 
                   np.array(obses_tp1), np.array(dones), mask)
        
        # Goal-conditioned sampling
        lg = min(len(self.goals), batch_size)
        goals_per_sample = max(1, batch_size // lg)
        
        obses_goal_t, actions, rewards, obses_goal_tp1, dones = [], [], [], [], []
        goal_indices = np.random.choice(len(self.goals), lg, replace=False)
        experience_indices = np.random.randint(0, len(self._storage), goals_per_sample)
        
        for i in range(batch_size):
            goal_idx = goal_indices[i % lg]
            exp_idx = experience_indices[i // lg % goals_per_sample]
            
            obs_t, action, reward, obs_tp1, done = self._storage[exp_idx]
            goal = self.goals[goal_idx]
            
            # Modify reward based on goal achievement
            if done and obs_t.sum() != goal.sum():
                reward = -2  # Penalty for not reaching the goal
            
            # Concatenate observation with goal
            obs_goal_t = np.concatenate((obs_t, goal), axis=2)
            obs_goal_tp1 = np.concatenate((obs_tp1, goal), axis=2)
            
            obses_goal_t.append(obs_goal_t)
            actions.append(action)
            rewards.append(reward)
            obses_goal_tp1.append(obs_goal_tp1)
            dones.append(done)
        
        # Create bootstrap mask
        if head_idx is not None:
            mask = (torch.rand(batch_size) < 0.5).float()
        else:
            mask = torch.ones(batch_size)
            
        return (np.array(obses_goal_t), np.array(actions), np.array(rewards), 
               np.array(obses_goal_tp1), np.array(dones), mask)


class BootstrappedDQN(nn.Module):
    def __init__(self, n_action, num_heads=10, shared_encoder=True):
        super(BootstrappedDQN, self).__init__()
        self.n_action = n_action
        self.num_heads = num_heads
        self.shared_encoder = shared_encoder

        # Shared encoder
        self.conv1 = nn.Conv2d(3+3, 32, kernel_size=8, stride=4)  # obs + goal
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear1 = nn.Linear(3136, 512)

        # Bootstrap heads
        self.heads = nn.ModuleList([
            nn.Linear(512, self.n_action) for _ in range(num_heads)
        ])

    def forward(self, x, head_idx=None):
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.linear1(x.reshape(x.size(0), -1)))
        
        if head_idx is not None:
            # Use specific head
            x = self.heads[head_idx](x)
        else:
            # Use all heads and return mean
            outputs = [head(x) for head in self.heads]
            x = torch.stack(outputs).mean(0)
        # Do NOT squeeze here; let caller handle shape
        return x

    def get_head_q_values(self, x, head_idx):
        """Get Q-values from specific head"""
        return self.forward(x, head_idx)


class BootstrappedGoalConditionedAgent(object):
    def __init__(self,
                 env,
                 training_stats={"R": [], "T": 0, "head_losses": []},
                 max_timesteps=2000000,
                 learning_starts=10000,
                 train_freq=4,
                 target_update_freq=1000,
                 learning_rate=1e-4,
                 batch_size=128,
                 replay_buffer_size=300000,
                 num_heads=10,
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
        self.num_heads = num_heads
        self.gamma = gamma
        self.print_freq = print_freq
        self.path = path
        
        self.training_stats = {
            "R": [], 
            "T": 0, 
            "head_losses": [[] for _ in range(num_heads)],
            "q_values": [[] for _ in range(num_heads)],
            "exploration_rate": []
        }

        self.eps_schedule = LinearSchedule(eps_timesteps, eps_final, eps_initial)

        # Initialize networks
        self.q_func = BootstrappedDQN(self.env.action_space.n, num_heads)
        self.target_q_func = BootstrappedDQN(self.env.action_space.n, num_heads)
        self.target_q_func.load_state_dict(self.q_func.state_dict())

        if use_cuda:
            self.q_func.cuda()
            self.target_q_func.cuda()

        # Separate optimizers for each head (optional) or one for all
        self.optimizer = optim.Adam(self.q_func.parameters(), lr=learning_rate)
        
        # Alternatively, separate optimizers for each head:
        # self.optimizers = [
        #     optim.Adam(list(self.q_func.shared_parameters()) + 
        #               list(self.q_func.heads[i].parameters()), 
        #               lr=learning_rate)
        #     for i in range(num_heads)
        # ]
        
        self.replay_buffer = BootstrappedReplayBuffer(replay_buffer_size, num_heads)
        self.steps = 0
        self.current_head = 0  # For action selection

    def select_action(self, obs, training=True):
        sample = random.random()
        eps_threshold = self.eps_schedule(self.steps) if training else 0.01
        
        if sample > eps_threshold and len(self.replay_buffer.goals) > 0:
            obs = np.array(obs)
            obs = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0)
            
            # Sample a random goal
            goal_idx = random.randint(0, len(self.replay_buffer.goals) - 1)
            goal = torch.from_numpy(self.replay_buffer.goals[goal_idx]).type(FloatTensor).unsqueeze(0)
            obs_goal = torch.cat((obs, goal), dim=3)
            
            # Use current head for action selection
            with torch.no_grad():
                q_values = self.q_func.get_head_q_values(obs_goal, self.current_head)
                # Ensure q_values is at least 2D
                if q_values.dim() == 1:
                    q_values = q_values.unsqueeze(0)
                action = q_values.max(1)[1].reshape(1, 1)
            
            # Rotate head for next selection
            self.current_head = (self.current_head + 1) % self.num_heads
            return action
        else:
            sample_action = self.env.action_space.sample()
            return torch.IntTensor([[sample_action]])

    def train(self):
        obs = self.env.reset()
        # Initialize training stats
        self.training_stats["R"] = [0]
        self.training_stats["T"] = 0
        self.training_stats["head_losses"] = [[] for _ in range(self.num_heads)]

        for t in range(self.max_timesteps):
            action = self.select_action(obs)
            # Ensure action is a scalar for env.step
            action_scalar = int(action[0][0]) if hasattr(action, '__getitem__') else int(action)
            new_obs, reward, done, info = self.env.step(action_scalar)
            self.replay_buffer.add(obs, action.cpu() if hasattr(action, 'cpu') else action, reward, new_obs, done)
            obs = new_obs

            # Update episode reward
            self.training_stats["R"][-1] += reward

            if done:
                obs = self.env.reset()
                self.training_stats["R"].append(0)

            # Training phase: only sample if enough data in buffer
            if t > self.learning_starts and t % self.train_freq == 0:
                if len(self.replay_buffer) >= self.batch_size:
                    self._train_step(t)
                else:
                    pass  # Not enough samples to train

            # Target network update
            if t > self.learning_starts and t % self.target_update_freq == 0:
                self.target_q_func.load_state_dict(self.q_func.state_dict())
                self._save_model_and_stats(t)

            self.steps += 1

            # Logging
            if done and self.print_freq is not None and len(self.training_stats["R"]) % self.print_freq == 0:
                self._log_progress(t)

        self.training_stats["T"] = self.steps

    def _train_step(self, t):
        """Perform one training step for all bootstrap heads"""
        total_loss = 0
        
        for head_idx in range(self.num_heads):
            # Sample batch with bootstrap mask for this head
            (obs_batch, act_batch, rew_batch, 
             next_obs_batch, done_mask, bootstrap_mask) = self.replay_buffer.sample(
                self.batch_size, head_idx)
            
            # Convert to tensors
            obs_batch = Variable(torch.from_numpy(obs_batch).type(FloatTensor))
            act_batch = Variable(torch.from_numpy(act_batch).type(LongTensor))
            rew_batch = Variable(torch.from_numpy(rew_batch).type(FloatTensor))
            next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(FloatTensor))
            not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(FloatTensor)
            bootstrap_mask = Variable(bootstrap_mask.type(FloatTensor))
            
            if use_cuda:
                act_batch = act_batch.cuda()
                rew_batch = rew_batch.cuda()
                bootstrap_mask = bootstrap_mask.cuda()

            # Compute current Q values for this head
            current_q_values = self.q_func.get_head_q_values(obs_batch, head_idx)
            # Ensure current_q_values is at least 2D
            if current_q_values.dim() == 1:
                current_q_values = current_q_values.unsqueeze(0)
            current_q_values = current_q_values.gather(1, act_batch.squeeze(2)).squeeze()
            
            # Compute next Q values from target network
            with torch.no_grad():
                next_q_values = self.target_q_func.get_head_q_values(next_obs_batch, head_idx)
                # Ensure next_q_values is at least 2D
                if next_q_values.dim() == 1:
                    next_q_values = next_q_values.unsqueeze(0)
                next_max_q = next_q_values.max(1)[0]
                target_q_values = rew_batch + (self.gamma * next_max_q * not_done_mask)

            # Apply bootstrap mask
            masked_current_q = current_q_values * bootstrap_mask
            masked_target_q = target_q_values * bootstrap_mask
            
            # Compute loss only for masked samples
            loss = F.smooth_l1_loss(masked_current_q, masked_target_q, reduction='sum')
            loss = loss / (bootstrap_mask.sum() + 1e-8)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.q_func.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Store head-specific stats
            self.training_stats["head_losses"][head_idx].append(loss.item())
            self.training_stats["q_values"][head_idx].append(current_q_values.mean().item())

        self.training_stats["exploration_rate"].append(self.eps_schedule(t))

    def _save_model_and_stats(self, t):
        """Save model and training statistics"""
        if self.path:
            torch.save(self.q_func.state_dict(), self.path + 'bootstrapped_dqn_model.pth')
            dd.io.save('bootstrapped_training_stats.h5', self.training_stats)
            print(f"\nModel and training stats saved at step {t}\n")

    def _log_progress(self, t):
        """Log training progress"""
        num_episodes = len(self.training_stats["R"])
        mean_100ep_reward = round(np.mean(self.training_stats["R"][-101:-1]), 1)
        
        print("--------------------------------------------------------")
        print(f"steps {t}")
        print(f"episodes {num_episodes}")
        print(f"mean 100 episode reward {mean_100ep_reward}")
        print(f"% time spent exploring {int(100 * self.eps_schedule(t))}")
        
        # Log head-specific information
        if len(self.training_stats["head_losses"][0]) > 0:
            avg_losses = [np.mean(losses[-100:]) for losses in self.training_stats["head_losses"]]
            print(f"avg head losses: {[round(l, 4) for l in avg_losses]}")
        print("--------------------------------------------------------")

    def evaluate(self, num_episodes=10, render=False):
        """Evaluate the agent"""
        total_rewards = []
        success_rate = 0
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 1000:  # Limit steps per episode
                if render:
                    self.env.render()
                
                action = self.select_action(obs, training=False)
                obs, reward, done, info = self.env.step(int(action[0][0]))
                episode_reward += reward
                steps += 1
                
                if done and reward > 0:  # Assuming positive reward for success
                    success_rate += 1
            
            total_rewards.append(episode_reward)
        
        avg_reward = np.mean(total_rewards)
        success_rate = success_rate / num_episodes
        
        print(f"Evaluation over {num_episodes} episodes:")
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Success rate: {success_rate:.2f}")
        
        return avg_reward, success_rate

