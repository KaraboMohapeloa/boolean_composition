# Training Stats - What Gets Saved

## During Training (Intermediate Saves)

**File**: `models/bootstrapped_blue/bdqn_training_stats.h5`

**Saved every 1000 steps** (when target network is updated)

**Contents**:
```python
{
    "R": [reward_ep1, reward_ep2, ..., reward_epN],  # List of episode rewards
    "T": total_timesteps                              # Total timesteps so far
}
```

### Field Descriptions:

1. **"R"** - Episode Rewards (list)
   - Total cumulative reward for each completed episode
   - Length = number of episodes completed so far
   - Updated at the END of each episode
   - Used to track learning progress over time
   
   Example: `[-5.0, -3.2, -4.1, -2.8, ...]`

2. **"T"** - Total Timesteps (integer)
   - Total number of environment steps taken
   - Updated after training completes
   - Target: 2,000,000 steps
   
   Example: `57000` (at the checkpoint you saw)

### How Rewards are Tracked:

```python
# Start of episode
self.training_stats["R"].append(0)  # Initialize episode reward to 0

# Each step
self.training_stats["R"][-1] += reward  # Add step reward to current episode

# End of episode (when done=True)
# Current episode reward is finalized, next episode starts fresh
```

---

## After Training Completes (Final Save)

**File**: `exps_data/bootstrapped/prime_experiment_stats.h5`

**Saved once** when entire experiment completes (2M steps)

**Contents**:
```python
data_stats_bootstrapped = np.array([
    [agent.training_stats]  # shape: (num_runs, num_tasks)
])

# Where agent.training_stats = {"R": [...], "T": 2000000}
```

This is a numpy array containing the training_stats dictionary for each run and task combination.

---

## Additional Files Saved

### 1. Model Checkpoints (Every 1000 steps)
**File**: `models/bootstrapped_blue/model_bdqn.pth`
- PyTorch state dictionary with neural network weights
- Contains weights for all 10 bootstrap heads
- Can be loaded to resume training or evaluate

### 2. Final Model (After completion)
**File**: `models/bootstrapped_blue/model_bdqn_final.pth`
- Same as above but marked as the final trained model

### 3. Final Stats (After completion)
**File**: `models/bootstrapped_blue/bdqn_training_stats_final.h5`
- Same structure as intermediate stats
- Final complete version

---

## What You Can Do With This Data

### 1. Monitor Training Progress
```python
import deepdish as dd

stats = dd.io.load('models/bootstrapped_blue/bdqn_training_stats.h5')
episode_rewards = stats['R']
total_steps = stats['T']

print(f"Completed {len(episode_rewards)} episodes")
print(f"Progress: {total_steps:,} / 2,000,000 steps")
print(f"Mean last 100 episodes: {np.mean(episode_rewards[-100:]):.2f}")
```

### 2. Plot Learning Curves
```python
import matplotlib.pyplot as plt
import numpy as np

# Plot raw rewards
plt.plot(episode_rewards, alpha=0.3)

# Plot moving average
window = 100
moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
plt.plot(range(window-1, len(episode_rewards)), moving_avg)

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Learning Progress')
plt.show()
```

### 3. Compare Methods
Once both vanilla and bootstrapped experiments complete, `plots.py` will:
- Load both `exps_data/vanilla/prime_experiment_stats.h5` and `exps_data/bootstrapped/prime_experiment_stats.h5`
- Extract episode rewards from both
- Create comparison plots showing which method learns faster/better

---

## Current Status

Based on your earlier output showing 57,000 steps:

```
Total Episodes: ~1,200-1,500 (estimated)
Episode Rewards: Array of ~1,200-1,500 values
Total Steps: 57,000
Progress: 2.85% complete
```

However, the current saved file appears to only have initialization data, suggesting:
1. The training that reached 57k steps crashed before properly saving
2. OR the file was overwritten when trying to restart
3. The dependency issue (numpy/deepdish) is preventing proper save/load

---

## Why Plots.py Doesn't Work Yet

`plots.py` is looking for:
```
exps_data/bootstrapped/prime_experiment_stats.h5
```

This file is **only created when training fully completes** (reaches 2M steps and exits normally).

The intermediate file `models/bootstrapped_blue/bdqn_training_stats.h5` exists but:
- Contains partial training data
- Is in a different location than plots.py expects
- May be corrupted if training crashed mid-save

---

## To Get Plottable Results

1. **Fix the numpy/deepdish compatibility issue**
2. **Run training to completion** (2,000,000 steps)
3. **Wait for "EXPERIMENT COMPLETE" message**
4. **Then run** `plots.py` functions

The training will take several hours/days depending on your GPU.
