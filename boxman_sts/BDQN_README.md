# Bootstrapped DQN for Boxman Environment

## Overview

This implements **Bootstrapped Deep Q-Networks (BDQN)** for the Boxman visual goal-reaching task, extending the tabular BDQN from `four_rooms` to high-dimensional image observations.

## Key Features

### 1. **Thompson Sampling Exploration**
- Samples one bootstrap head uniformly at random
- Acts greedily with respect to selected head
- Natural exploration through posterior sampling
- More sample-efficient than ε-greedy

### 2. **Ensemble Architecture**
```python
BootstrappedDQN:
  - Shared convolutional features (conv1, conv2, conv3)
  - 10 separate fully-connected heads
  - Each head learns different Q-function approximation
  - Uncertainty = variance across heads
```

### 3. **Per-Step Masking**
- Each transition included in each head with probability `mask_prob=0.8`
- Creates diversity across bootstrap heads
- More sample-efficient than separate replay buffers

### 4. **Three-Phase Training**

**Phase 1: Warmup (5,000 steps)**
- Pure random exploration
- Collect initial dataset
- Store transitions in buffer

**Phase 2: Bootstrap**
- Train all heads on warmup data
- Per-step masking for diversity
- Initialize with good Q-values

**Phase 3: Learning**
- Thompson sampling for actions
- Per-step masked updates
- Continuous improvement

### 5. **Multiple Exploration Strategies**

#### Thompson Sampling (Recommended)
```python
exploration_strategy='thompson'
```
- Sample one head uniformly
- Act greedily w.r.t. selected head
- Best sample efficiency

#### Upper Confidence Bound
```python
exploration_strategy='ucb'
```
- UCB = mean(Q) + 2.0 * sqrt(var(Q))
- Optimistic exploration
- Good for sparse rewards

#### Ensemble Voting
```python
exploration_strategy='vote'
```
- Each head votes for best action
- Select majority action
- Conservative but stable

#### Ensemble Mean
```python
exploration_strategy='mean'
```
- Use averaged Q-values
- No exploration bonus
- Best for evaluation

## Architecture Details

### Network Structure
```
Input: 84×84×6 (state + goal concatenated)
  ↓
Conv1: 32 filters, 8×8 kernel, stride 4 → 20×20×32
  ↓
Conv2: 64 filters, 4×4 kernel, stride 2 → 9×9×64
  ↓
Conv3: 64 filters, 3×3 kernel, stride 1 → 7×7×64
  ↓
Flatten: 3136 features
  ↓
[Head 1] → FC1(512) → FC2(n_actions)
[Head 2] → FC1(512) → FC2(n_actions)
...
[Head 10] → FC1(512) → FC2(n_actions)
```

### Goal-Conditioned Learning
- Input: concatenate state + goal images
- Replay buffer stores discovered goals
- Hindsight Experience Replay (HER)-like relabeling
- Penalty for reaching wrong terminal states

## Usage

### Training

```python
from bdqn import BootstrappedAgent
from gym_repoman.envs import CollectEnv
from wrappers import WarpFrame

# Create environment
env = WarpFrame(CollectEnv(goal_condition=lambda x: x.colour == 'blue'))

# Create agent
agent = BootstrappedAgent(
    env=env,
    path='./models/bdqn_blue/',
    n_heads=10,                    # Number of bootstrap heads
    mask_prob=0.8,                 # Per-step masking probability
    init_q_range=0.2,              # Optimistic initialization
    warmup_steps=5000,             # Random exploration phase
    exploration_strategy='thompson', # Thompson sampling
    max_timesteps=2000000,
    batch_size=128,
    learning_rate=1e-4,
    gamma=0.99
)

# Train
agent.train()
```

### Evaluation

```python
# Load trained model
agent.q_func.load_state_dict(torch.load('model_bdqn.pth'))

# Evaluate with ensemble mean (no exploration)
agent.exploration_strategy = 'mean'
total_reward = 0
for episode in range(100):
    obs = env.reset()
    done = False
    while not done:
        action = agent.select_action(obs)
        obs, reward, done, _ = env.step(int(action[0][0]))
        total_reward += reward
print(f"Average reward: {total_reward / 100}")
```

## Running Experiments

### Compare BDQN vs Vanilla DQN

```bash
python prime_experiment_bdqn.py
```

This trains both vanilla DQN and Bootstrapped DQN on the same tasks and saves:
- Model checkpoints
- Training statistics
- Comparison plots

### Results Location
```
models/
  dqn_blue/          # Vanilla DQN models
  bdqn_thompson_blue/ # Bootstrapped DQN models
  
exps_data/
  dqn/
    comparison_stats.h5
  bdqn/
    comparison_stats.h5
```

## Expected Improvements

### Sample Efficiency
- **~30-50% fewer episodes** to reach same performance
- Faster convergence through better exploration
- More stable learning curves

### Exploration Quality
- Better coverage of state space
- Natural exploration without manual tuning
- Adapts to uncertainty

### Robustness
- Multiple heads provide redundancy
- Uncertainty quantification
- More reliable policies

## Comparison with Four Rooms BDQN

### Similarities
✅ Thompson sampling for exploration
✅ Per-step masking (mask_prob=0.8)
✅ Ensemble of Q-functions
✅ Warmup + bootstrap + learning phases
✅ Policy agreement convergence criteria

### Differences
📊 **State representation**: Images vs discrete positions
🧠 **Function approximation**: Neural networks vs tables
💾 **Memory**: Replay buffer vs direct updates
🎯 **Goals**: Visual objects vs grid positions
⚡ **Scalability**: High-dimensional vs tabular

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_heads` | 10 | Number of bootstrap heads |
| `mask_prob` | 0.8 | Probability of including transition in head |
| `init_q_range` | 0.2 | Optimistic initialization range |
| `warmup_steps` | 5000 | Random exploration steps |
| `exploration_strategy` | 'thompson' | Action selection method |
| `batch_size` | 128 | Training batch size |
| `learning_rate` | 1e-4 | Adam optimizer learning rate |
| `gamma` | 0.99 | Discount factor |

## Uncertainty Analysis

### Get Uncertainty Estimates
```python
obs = torch.from_numpy(obs).float()
uncertainty = agent.q_func.get_uncertainty(obs)
# High uncertainty = explore more
# Low uncertainty = exploit
```

### Ensemble Statistics
```python
# Get Q-values from all heads
all_q_values = agent.q_func.get_ensemble_q_values(obs)
# Shape: (n_heads, batch_size, n_actions)

# Mean Q-values
mean_q = agent.q_func.get_mean_q_values(obs)

# Variance (uncertainty)
var_q = agent.q_func.get_uncertainty(obs)
```

## Boolean Task Composition

The BDQN supports the same boolean operations as vanilla DQN:

```python
from bdqn import ComposedDQN

# Learn primitive tasks
bdqn_blue = BootstrappedDQN(n_actions=4, n_heads=10)
bdqn_circle = BootstrappedDQN(n_actions=4, n_heads=10)

# Compose with boolean operations
bdqn_blue_OR_circle = ComposedDQN([bdqn_blue, bdqn_circle], compose='or')
bdqn_blue_AND_circle = ComposedDQN([bdqn_blue, bdqn_circle], compose='and')
bdqn_NOT_blue = ComposedDQN([bdqn_blue], compose='not')
```

## Debugging

### Check Head Updates
```python
print(f"Head update counts: {agent.update_counts}")
# Should be roughly balanced across heads
```

### Monitor Exploration
```python
# During training, print which head is selected
head_idx = np.random.randint(agent.n_heads)
print(f"Thompson sampling selected head {head_idx}")
```

### Visualize Uncertainty
```python
import matplotlib.pyplot as plt

obs = env.reset()
obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

# Get Q-values from all heads
all_q = agent.q_func.get_ensemble_q_values(obs_tensor)

# Plot
plt.figure(figsize=(10, 6))
for action in range(env.action_space.n):
    q_values = all_q[:, 0, action].detach().cpu().numpy()
    plt.scatter([action]*len(q_values), q_values, alpha=0.6)
plt.xlabel('Action')
plt.ylabel('Q-value')
plt.title('Q-value Distribution Across Bootstrap Heads')
plt.show()
```

## Citation

If you use this implementation, please cite:

```
Osband et al. "Deep Exploration via Bootstrapped DQN" NIPS 2016
Barreto et al. "A Boolean Task Algebra for Reinforcement Learning" 2020
```

## Files

- `bdqn.py` - Main Bootstrapped DQN implementation
- `prime_experiment_bdqn.py` - Training script comparing DQN vs BDQN
- `BDQN_README.md` - This file

## Contact

For questions or issues, please open an issue on the repository.
