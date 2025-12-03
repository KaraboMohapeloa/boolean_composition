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


class BootstrappedReplayBuffer(object):
    """
    Bootstrapped replay buffer with per-step masking.
    Each transition is independently sampled for each head during training.
    """
    def __init__(self, size, n_heads=10, mask_prob=0.8, N=-100):
        """
        Create Bootstrapped Replay buffer.
        
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer.
        n_heads: int
            Number of bootstrap heads
        mask_prob: float
            Probability of including each transition in each head (per-step masking)
        N: float
            Penalty for reaching wrong terminal state
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.n_heads = n_heads
        self.mask_prob = mask_prob
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
        """
        Add transition to buffer.
        Per-step masking: generate mask for each head at storage time.
        """
        data = (obs_t, action, reward, obs_tp1, done)
            
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize 

    def sample(self, batch_size):
        """
        Sample a batch of experiences for standard (non-bootstrap) training.
        Used during warmup phase.
        
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
            
        Returns
        -------
        Batch of transitions with goal augmentation
        """
        obses_goal_t, actions, rewards, obses_goal_tp1, dones = [], [], [], [], []
        lg = len(self.goals)
        
        if lg == 0:
            # No goals yet, return empty batch
            return (np.array(obses_goal_t), np.array(actions), np.array(rewards), 
                    np.array(obses_goal_tp1), np.array(dones))
        
        ng = np.arange(lg)
        np.random.shuffle(ng)  
        mbs = int(batch_size / lg)
        indices = np.random.randint(0, len(self._storage), mbs)             
                
        for i in range(batch_size):
            obs_t, action, reward, obs_tp1, done = self._storage[indices[i % mbs]] 
            obs_t = np.array(obs_t, copy=False)
            obs_tp1 = np.array(obs_tp1, copy=False)               
            
            goal = self.goals[ng[int(i / mbs) % lg]]
            if done and obs_t.sum() != goal.sum():
                reward = self.N   # Penalty for wrong terminal state
            
            obses_goal_t.append(np.concatenate((obs_t, goal), axis=2))
            actions.append(np.array(action.cpu(), copy=False))
            rewards.append(reward)
            obses_goal_tp1.append(np.concatenate((obs_tp1, goal), axis=2))
            dones.append(done)
            
        return (np.array(obses_goal_t), np.array(actions), np.array(rewards), 
                np.array(obses_goal_tp1), np.array(dones))

    def sample_bootstrapped(self, batch_size, head_idx):
        """
        Sample a batch for a specific bootstrap head with per-step masking.
        
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        head_idx: int
            Which bootstrap head to sample for
            
        Returns
        -------
        Batch of transitions with goal augmentation and bootstrap masking
        """
        obses_goal_t, actions, rewards, obses_goal_tp1, dones = [], [], [], [], []
        lg = len(self.goals)
        
        if lg == 0:
            return (np.array(obses_goal_t), np.array(actions), np.array(rewards), 
                    np.array(obses_goal_tp1), np.array(dones))
        
        ng = np.arange(lg)
        np.random.shuffle(ng)  
        mbs = int(batch_size / lg)
        
        # Sample transitions
        sampled = 0
        attempts = 0
        max_attempts = batch_size * 10  # Prevent infinite loop
        
        while sampled < batch_size and attempts < max_attempts:
            idx = np.random.randint(0, len(self._storage))
            attempts += 1
            
            # Per-step masking: include this transition with probability mask_prob
            if np.random.random() < self.mask_prob:
                obs_t, action, reward, obs_tp1, done = self._storage[idx]
                obs_t = np.array(obs_t, copy=False)
                obs_tp1 = np.array(obs_tp1, copy=False)
                
                goal = self.goals[ng[sampled % lg]]
                if done and obs_t.sum() != goal.sum():
                    reward = self.N
                
                obses_goal_t.append(np.concatenate((obs_t, goal), axis=2))
                # Convert action tensor to numpy and flatten to scalar for batching
                action_np = action.cpu().numpy() if hasattr(action, 'cpu') else np.array(action)
                actions.append(action_np.item() if action_np.size == 1 else action_np.flatten()[0])
                rewards.append(reward)
                obses_goal_tp1.append(np.concatenate((obs_tp1, goal), axis=2))
                dones.append(done)
                sampled += 1
        
        # If we couldn't sample enough, fall back to standard sampling
        if sampled < batch_size:
            return self.sample(batch_size)
            
        return (np.array(obses_goal_t), np.array(actions), np.array(rewards), 
                np.array(obses_goal_tp1), np.array(dones))


class BootstrappedDQN(nn.Module):
    """
    Bootstrapped DQN with multiple heads for uncertainty estimation.
    Each head shares convolutional features but has separate fully-connected layers.
    """
    def __init__(self, n_action, n_heads=10, shared_features=True):
        super(BootstrappedDQN, self).__init__()
        self.n_action = n_action
        self.n_heads = n_heads
        self.shared_features = shared_features

        # Shared convolutional layers (feature extraction)
        self.conv1 = nn.Conv2d(3+3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Multiple bootstrap heads
        if shared_features:
            # Shared conv features, separate FC layers for each head
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(3136, 512),
                    nn.ReLU(),
                    nn.Linear(512, self.n_action)
                ) for _ in range(n_heads)
            ])
        else:
            # Completely separate heads (more diversity but more parameters)
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(3136, 512),
                    nn.ReLU(),
                    nn.Linear(512, self.n_action)
                ) for _ in range(n_heads)
            ])

    def forward(self, x, head_idx=None):
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x: tensor
            Input observation (state + goal)
        head_idx: int or None
            If specified, return Q-values for specific head.
            If None, return Q-values for all heads.
        
        Returns
        -------
        Q-values for specified head(s)
        """
        x = x.permute(0, 3, 1, 2)
        
        # Shared feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        
        # Head-specific processing
        if head_idx is not None:
            # Single head
            return self.heads[head_idx](x)
        else:
            # All heads
            outputs = [head(x) for head in self.heads]
            return torch.stack(outputs, dim=0)  # Shape: (n_heads, batch_size, n_actions)

    def get_ensemble_q_values(self, x):
        """
        Get Q-values from all heads for ensemble methods.
        
        Returns
        -------
        q_values: tensor (n_heads, batch_size, n_actions)
        """
        return self.forward(x, head_idx=None)

    def get_mean_q_values(self, x):
        """
        Get ensemble-averaged Q-values.
        
        Returns
        -------
        mean_q: tensor (batch_size, n_actions)
        """
        q_values = self.get_ensemble_q_values(x)
        return q_values.mean(dim=0)

    def get_uncertainty(self, x):
        """
        Get uncertainty (variance) in Q-value estimates across heads.
        
        Returns
        -------
        uncertainty: tensor (batch_size, n_actions)
        """
        q_values = self.get_ensemble_q_values(x)
        return q_values.var(dim=0)

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        # copy params
        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param)
        # freeze params
        for name, param in self.named_parameters():
            if name.split(".")[0] in ["conv1", "conv2", "conv3"]:
                param.requires_grad = False


class BootstrappedAgent(object):
    """
    Bootstrapped DQN Agent with Thompson Sampling for exploration.
    Implements the core BDQN algorithm from the four_rooms implementation.
    """
    def __init__(self,
                 env,
                 training_stats={"R": [], "T": 0},
                 max_timesteps=2000000,
                 learning_starts=10000,
                 warmup_steps=5000,
                 train_freq=4,
                 target_update_freq=1000,
                 learning_rate=1e-4,
                 batch_size=128,
                 replay_buffer_size=300000,
                 gamma=0.99,
                 n_heads=10,
                 mask_prob=0.8,
                 print_freq=10,
                 exploration_strategy='thompson',
                 path=None):
        """
        Initialize Bootstrapped DQN Agent.
        
        Parameters
        ----------
        exploration_strategy: str
            'thompson': Thompson sampling (sample one head)
            'vote': Ensemble voting
            'ucb': Upper confidence bound
            'mean': Use ensemble mean (no exploration bonus)
        """
        assert type(env.observation_space) == gym.spaces.Box
        assert type(env.action_space) == gym.spaces.Discrete

        self.env = env
        self.max_timesteps = max_timesteps
        self.learning_starts = learning_starts
        self.warmup_steps = warmup_steps
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.gamma = gamma
        self.print_freq = print_freq
        self.path = path
        self.n_heads = n_heads
        self.mask_prob = mask_prob
        self.exploration_strategy = exploration_strategy
        self.training_stats = {"R": [], "T": 0}

        # Bootstrapped Q-networks
        self.q_func = BootstrappedDQN(self.env.action_space.n, n_heads=n_heads)
        self.target_q_func = BootstrappedDQN(self.env.action_space.n, n_heads=n_heads)
        self.target_q_func.load_state_dict(self.q_func.state_dict())

        if use_cuda:
            self.q_func.cuda()
            self.target_q_func.cuda()

        # Separate optimizers for each head (or single optimizer for all)
        self.optimizer = optim.Adam(self.q_func.parameters(), lr=learning_rate)
        
        # Bootstrapped replay buffer
        self.replay_buffer = BootstrappedReplayBuffer(
            replay_buffer_size, 
            n_heads=n_heads,
            mask_prob=mask_prob,
            N=(env.rmin - env.rmax) * env.diameter if hasattr(env, 'diameter') else -100
        )
        
        self.steps = 0
        self.update_counts = [0] * n_heads  # Track updates per head

    def set_intermediate_save_callback(self, callback):
        """
        Register a callback to be called every time an intermediate save occurs.
        The callback will be called as: callback(self.training_stats)
        """
        self._intermediate_save_callback = callback

    def select_action_thompson(self, obs):
        """
        Thompson Sampling: Randomly select one head and act greedily with respect to it.
        This is the core exploration strategy from Bootstrapped DQN.
        """
        if len(self.replay_buffer.goals) == 0:
            return torch.IntTensor([[self.env.action_space.sample()]])
        
        obs = np.array(obs)
        obs = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0)
        
        # Thompson sampling: sample one head uniformly at random
        head_idx = np.random.randint(self.n_heads)
        
        values = []
        for goal in self.replay_buffer.goals:
            goal = torch.from_numpy(np.array(goal)).type(FloatTensor).unsqueeze(0)
            x = torch.cat((obs, goal), dim=3)
            with torch.no_grad():
                # Get Q-values from selected head only
                q_vals = self.q_func(x, head_idx=head_idx).squeeze(0)
                values.append(q_vals)
        
        # Generalized Policy Improvement: max over goals
        values = torch.stack(values, 1).t()
        action = values.data.max(0)[0].max(0)[1].reshape(1, 1)
        return action

    def select_action_vote(self, obs):
        """
        Ensemble Voting: Each head votes for best action, select majority.
        """
        if len(self.replay_buffer.goals) == 0:
            return torch.IntTensor([[self.env.action_space.sample()]])
        
        obs = np.array(obs)
        obs = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0)
        
        votes = torch.zeros(self.env.action_space.n)
        
        for goal in self.replay_buffer.goals:
            goal = torch.from_numpy(np.array(goal)).type(FloatTensor).unsqueeze(0)
            x = torch.cat((obs, goal), dim=3)
            
            with torch.no_grad():
                # Get Q-values from all heads
                all_q_vals = self.q_func.get_ensemble_q_values(x)  # (n_heads, 1, n_actions)
                
                # Each head votes for its best action
                for head_idx in range(self.n_heads):
                    best_action = all_q_vals[head_idx].squeeze(0).max(0)[1]
                    votes[best_action] += 1
        
        # Select action with most votes
        action = votes.max(0)[1].reshape(1, 1)
        return action

    def select_action_ucb(self, obs, optimism_factor=2.0):
        """
        Upper Confidence Bound: Select action with highest optimistic Q-value.
        Q_ucb = mean(Q) + optimism_factor * sqrt(var(Q))
        """
        if len(self.replay_buffer.goals) == 0:
            return torch.IntTensor([[self.env.action_space.sample()]])
        
        obs = np.array(obs)
        obs = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0)
        
        ucb_values = []
        for goal in self.replay_buffer.goals:
            goal = torch.from_numpy(np.array(goal)).type(FloatTensor).unsqueeze(0)
            x = torch.cat((obs, goal), dim=3)
            
            with torch.no_grad():
                mean_q = self.q_func.get_mean_q_values(x).squeeze(0)
                uncertainty = self.q_func.get_uncertainty(x).squeeze(0)
                # UCB = mean + optimism * sqrt(variance)
                ucb = mean_q + optimism_factor * torch.sqrt(uncertainty + 1e-8)
                ucb_values.append(ucb)
        
        # GPI: max over goals
        ucb_values = torch.stack(ucb_values, 1).t()
        action = ucb_values.data.max(0)[0].max(0)[1].reshape(1, 1)
        return action

    def select_action_mean(self, obs):
        """
        Ensemble Mean: Use averaged Q-values (no exploration bonus).
        """
        if len(self.replay_buffer.goals) == 0:
            return torch.IntTensor([[self.env.action_space.sample()]])
        
        obs = np.array(obs)
        obs = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0)
        
        values = []
        for goal in self.replay_buffer.goals:
            goal = torch.from_numpy(np.array(goal)).type(FloatTensor).unsqueeze(0)
            x = torch.cat((obs, goal), dim=3)
            with torch.no_grad():
                mean_q = self.q_func.get_mean_q_values(x).squeeze(0)
                values.append(mean_q)
        
        values = torch.stack(values, 1).t()
        action = values.data.max(0)[0].max(0)[1].reshape(1, 1)
        return action

    def select_action(self, obs):
        """
        Select action based on configured exploration strategy.
        """
        if self.exploration_strategy == 'thompson':
            return self.select_action_thompson(obs)
        elif self.exploration_strategy == 'vote':
            return self.select_action_vote(obs)
        elif self.exploration_strategy == 'ucb':
            return self.select_action_ucb(obs)
        elif self.exploration_strategy == 'mean':
            return self.select_action_mean(obs)
        else:
            # Fallback to Thompson sampling
            return self.select_action_thompson(obs)

    def train(self):
        """
        Main training loop implementing Bootstrapped DQN algorithm.
        
        Phase 1: Warmup - Random exploration to collect initial dataset
        Phase 2: Bootstrap - Train all heads on collected data with masking
        Phase 3: Learning - Thompson sampling + per-step masked updates
        """
        obs = self.env.reset()
        self.training_stats["R"] = []
        self.training_stats["T"] = 0
        self.training_stats["R"].append(0)
        
        warmup_buffer = []
        in_warmup = True

        for t in range(self.max_timesteps):
            # Phase 1: Warmup with pure random exploration
            if t < self.warmup_steps:
                action = torch.IntTensor([[self.env.action_space.sample()]])
            else:
                # Phase 3: Thompson sampling exploration
                if in_warmup:
                    print(f"\n=== Warmup Complete ({self.warmup_steps} steps) ===")
                    print(f"Collected {len(warmup_buffer)} transitions for bootstrap")
                    print("Starting bootstrap phase...")
                    
                    # Phase 2: Bootstrap all heads with warmup data
                    self._bootstrap_from_warmup(warmup_buffer)
                    warmup_buffer = []  # Free memory
                    in_warmup = False
                    print("Bootstrap complete. Starting Thompson sampling...\n")
                
                action = self.select_action(obs)

            # Execute action
            new_obs, reward, done, info = self.env.step(int(action[0][0]))
            self.replay_buffer.add(obs, action.cpu(), reward, new_obs, done)
            obs = new_obs

            # Track rewards
            self.training_stats["R"][-1] += reward

            # Store warmup transitions
            if t < self.warmup_steps:
                warmup_buffer.append((obs, action, reward, new_obs, done))

            # --- Prevent getting stuck in one episode forever ---
            max_episode_steps = 20000  # or another reasonable value
            if 'episode_step' not in locals():
                episode_step = 0
            episode_step += 1
            if episode_step >= max_episode_steps:
                done = True
                print(f"[WARNING] Forcing episode end at {max_episode_steps} steps to prevent infinite episode.")

            if done:
                obs = self.env.reset()
                self.training_stats["R"].append(0)
                episode_step = 0

            # Training updates (after warmup AND after learning_starts)
            if t >= self.warmup_steps and t >= self.learning_starts and t % self.train_freq == 0:
                # Train each head with per-step masking
                for head_idx in range(self.n_heads):
                    batch = self.replay_buffer.sample_bootstrapped(self.batch_size, head_idx)
                    
                    if len(batch[0]) > 0:  # Check if we got valid samples
                        self._train_head(head_idx, batch)

            # Update target network
            if t > self.warmup_steps and t % self.target_update_freq == 0:
                self.target_q_func.load_state_dict(self.q_func.state_dict())
                if self.path:
                    try:
                        self.training_stats["T"] = t
                        torch.save(self.q_func.state_dict(), self.path + 'model_bdqn.pth')
                        dd.io.save(self.path + 'bdqn_training_stats.h5', self.training_stats)
                        print(f"\nModel and stats saved (step {t})")
                        print(f"  Model: {self.path}model_bdqn.pth")
                        print(f"  Stats: {self.path}bdqn_training_stats.h5")
                        print(f"  Episodes: {len(self.training_stats['R'])}")
                        print(f"Head update counts: {self.update_counts}")
                        # Call the callback if set
                        if hasattr(self, '_intermediate_save_callback') and self._intermediate_save_callback:
                            self._intermediate_save_callback(self.training_stats)
                    except Exception as e:
                        print(f"\nWARNING: Failed to save at step {t}: {e}")
                        import traceback
                        traceback.print_exc()

            self.steps += 1

            # Logging
            rewards_window = self.training_stats["R"][-101:-1]
            if len(rewards_window) > 0:
                mean_100ep_reward = round(np.mean(rewards_window), 1)
            else:
                mean_100ep_reward = 0
            num_episodes = len(self.training_stats["R"])

            # Print average reward for last 100 episodes every print_freq episodes
            if done and self.print_freq is not None and num_episodes % self.print_freq == 0:
                print("--------------------------------------------------------")
                print(f"steps: {t}")
                print(f"episodes: {num_episodes}")
                print(f"mean 100 episode reward: {mean_100ep_reward}")
                print(f"exploration strategy: {self.exploration_strategy}")
                print(f"known goals: {len(self.replay_buffer.goals)}")
                print(f"head updates: {self.update_counts}")
                print("--------------------------------------------------------")

            # Always print average reward for last 100 episodes every 100 episodes
            if done and num_episodes % 100 == 0:
                print(f"[INFO] Average reward (last 100 episodes): {mean_100ep_reward}")

            # Print average reward for last 1000 steps every 1000 steps
            if t > 0 and t % 1000 == 0:
                # Find which episodes cover the last 1000 steps
                steps_per_episode = self.training_stats["T"] / max(1, len(self.training_stats["R"]))
                episodes_approx = int(1000 / steps_per_episode) if steps_per_episode > 0 else 1
                rewards_window_1000 = self.training_stats["R"][-episodes_approx:]
                mean_1000steps_reward = round(np.mean(rewards_window_1000), 2) if rewards_window_1000 else 0
                print(f"[INFO] Average reward (last ~1000 steps): {mean_1000steps_reward}")

        # Final save (use max_timesteps or self.steps, whichever is accurate)
        self.training_stats["T"] = self.steps
        if self.path:
            torch.save(self.q_func.state_dict(), self.path + 'model_bdqn_final.pth')
            dd.io.save(self.path + 'bdqn_training_stats_final.h5', self.training_stats)
            print(f"\n{'='*70}")
            print(f"TRAINING COMPLETE")
            print(f"{'='*70}")
            print(f"Total timesteps: {self.steps:,}")
            print(f"Total episodes: {len(self.training_stats['R'])}")
            print(f"Final model saved: {self.path}model_bdqn_final.pth")
            print(f"Final stats saved: {self.path}bdqn_training_stats_final.h5")
            print(f"{'='*70}\n")

    def _bootstrap_from_warmup(self, warmup_buffer):
        """
        Bootstrap phase: Train all heads on warmup data with per-step masking.
        Each transition is independently sampled for each head.
        """
        for head_idx in range(self.n_heads):
            head_updates = 0
            
            for transition in warmup_buffer:
                # Per-step masking: include with probability mask_prob
                if np.random.random() < self.mask_prob:
                    obs, action, reward, new_obs, done = transition
                    
                    # Goal augmentation
                    for goal in self.replay_buffer.goals:
                        obs_np = np.array(obs, copy=False)
                        new_obs_np = np.array(new_obs, copy=False)
                        goal_np = np.array(goal, copy=False)
                        
                        # Reward relabeling
                        goal_reward = reward
                        if done and obs_np.sum() != goal_np.sum():
                            goal_reward = self.replay_buffer.N
                        
                        # Create batch of 1
                        obs_goal = np.concatenate((obs_np, goal_np), axis=2)
                        new_obs_goal = np.concatenate((new_obs_np, goal_np), axis=2)
                        
                        batch = (
                            np.array([obs_goal]),
                            action.cpu().numpy(),  # Already has shape (1, 1), don't wrap again
                            np.array([goal_reward]),
                            np.array([new_obs_goal]),
                            np.array([done])
                        )
                        
                        self._train_head(head_idx, batch)
                        head_updates += 1
            
            self.update_counts[head_idx] = head_updates
            print(f"Head {head_idx}: {head_updates} bootstrap updates")

    def _train_head(self, head_idx, batch):
        """
        Train a specific bootstrap head with given batch.
        
        Parameters
        ----------
        head_idx: int
            Index of head to train
        batch: tuple
            (obs_batch, act_batch, rew_batch, next_obs_batch, done_mask)
        """
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = batch
        
        obs_batch = Variable(torch.from_numpy(obs_batch).type(FloatTensor))
        # Reshape actions to (batch_size, 1) for gather operation
        act_batch = Variable(torch.from_numpy(act_batch).type(LongTensor)).unsqueeze(1)
        rew_batch = Variable(torch.from_numpy(rew_batch).type(FloatTensor))
        next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(FloatTensor))
        not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(FloatTensor)

        if use_cuda:
            act_batch = act_batch.cuda()
            rew_batch = rew_batch.cuda()

        # Current Q-values from this head
        current_q_values = self.q_func(obs_batch, head_idx=head_idx).gather(1, act_batch).squeeze(1)
        
        # Target Q-values from this head's target network
        with torch.no_grad():
            next_max_q = self.target_q_func(next_obs_batch, head_idx=head_idx).detach().max(1)[0]
        next_q_values = not_done_mask * next_max_q
        target_q_values = rew_batch + (self.gamma * next_q_values)

        # Huber loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        for params in self.q_func.parameters():
            if params.grad is not None:
                params.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()
        self.update_counts[head_idx] += 1


# Boolean composition classes (same as original DQN)
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
        if self.compose == "or":
            q = qs.max(0)[0]
        elif self.compose == "and":
            q = qs.min(0)[0]
        else:  # not
            q_max = self.dqn_max(obs_goal)
            q_min = q_max - (self.rmax - self.rmin)
            q = (q_max + q_min) - qs[0]

        return q.detach().clone()


class MaxDQN(nn.Module):
    def __init__(self, dqn, rmax=2):
        super(MaxDQN, self).__init__()
        self.dqn = dqn
        self.rmax = rmax
    
    def forward(self, obs_goal):
        dqn_max = self.dqn(obs_goal)
        s = obs_goal[:, :, :, :3]
        g = obs_goal[:, :, :, 3:]        
        if s.sum() != g.sum():
            q_gg = self.dqn(torch.cat((g, g), dim=3))
            c = self.rmax - q_gg.max()
            dqn_max = dqn_max + c
        else:
            dqn_max = dqn_max * 0 + self.rmax        
        return dqn_max
