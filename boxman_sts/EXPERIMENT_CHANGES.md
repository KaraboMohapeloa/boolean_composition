# Prime Experiment: Vanilla DQN vs Bootstrapped DQN

## Overview

The `prime_experiment.py` script has been updated to compare **Vanilla DQN** (epsilon-greedy) with **Bootstrapped DQN** (Thompson sampling) instead of the previous softmax DQN comparison.

## Changes Made

### 1. **Replaced Softmax DQN with Bootstrapped DQN**

**Before:**
- Vanilla DQN (ε-greedy) vs Softmax DQN

**After:**
- Vanilla DQN (ε-greedy) vs Bootstrapped DQN (Thompson sampling)

### 2. **Why This Comparison?**

| Aspect | Vanilla DQN | Bootstrapped DQN |
|--------|-------------|------------------|
| **Exploration** | ε-greedy (random 10% of time) | Thompson sampling (uncertainty-driven) |
| **Q-function** | Single network | Ensemble of 10 heads |
| **Sample efficiency** | Baseline | 30-50% improvement expected |
| **Uncertainty** | None | Variance across ensemble |
| **Computational cost** | Lower | ~2-3x higher |

## File Structure

```
boxman_sts/
├── dqn.py                    # Vanilla DQN implementation
├── bdqn.py                   # NEW: Bootstrapped DQN implementation
├── prime_experiment.py       # UPDATED: Vanilla vs BDQN comparison
├── analyze_results.py        # NEW: Results analysis script
├── BDQN_README.md           # NEW: BDQN documentation
├── exps_data/
│   ├── vanilla/             # Vanilla DQN results
│   │   └── prime_experiment_stats.h5
│   └── bootstrapped/        # NEW: Bootstrapped DQN results
│       └── prime_experiment_stats.h5
└── models/
    ├── vanilla_{task}/      # Vanilla DQN models
    └── bootstrapped_{task}/ # NEW: Bootstrapped DQN models
```

## Running the Experiment

### 1. Train Both Methods

```bash
python prime_experiment.py
```

This will:
- Train Vanilla DQN with ε-greedy exploration
- Train Bootstrapped DQN with Thompson sampling (10 heads)
- Save results to `exps_data/vanilla/` and `exps_data/bootstrapped/`
- Print comparison summary after each task

### 2. Analyze Results

```bash
python analyze_results.py
```

This generates:
- **Sample efficiency analysis**: Episodes to reach 50%, 75%, 90% of final performance
- **Learning curves**: Comparative plots
- **Comparison table**: Markdown summary

## Expected Results

### Sample Efficiency Gains

Based on the BDQN algorithm, we expect:

1. **30-50% fewer episodes** to reach same performance level
2. **Faster convergence** in early training (better exploration)
3. **Similar or better final performance**
4. **More stable learning** (smoother curves)

### Example Output

```
TASK blue SUMMARY
======================================================================
Vanilla DQN      - Final 100-ep avg: 0.85
Bootstrapped DQN - Final 100-ep avg: 0.92
BDQN improvement: +0.07 (+8.2%)
======================================================================

SAMPLE EFFICIENCY (Episodes to reach performance thresholds):
  50% of final performance (0.46 reward):
    Vanilla: 1250 episodes
    BDQN:     850 episodes
    Gain:    +32.0% fewer episodes ✅
  
  75% of final performance (0.69 reward):
    Vanilla: 2100 episodes
    BDQN:    1450 episodes
    Gain:    +31.0% fewer episodes ✅
  
  90% of final performance (0.83 reward):
    Vanilla: 3500 episodes
    BDQN:    2400 episodes
    Gain:    +31.4% fewer episodes ✅
```

## Key Parameters

### Vanilla DQN
```python
- exploration: ε-greedy (ε: 1.0 → 0.01 over 1M steps)
- learning_rate: 1e-4
- batch_size: 128
- replay_buffer: 300,000
```

### Bootstrapped DQN
```python
- exploration: Thompson sampling
- n_heads: 10
- mask_prob: 0.8 (per-step masking)
- init_q_range: 0.2 (optimistic initialization)
- warmup_steps: 5,000
- learning_rate: 1e-4
- batch_size: 128
- replay_buffer: 300,000
```

## Metrics Tracked

Both methods track:
- `R`: Reward per episode
- `T`: Total timesteps
- Episode count
- Training time

Bootstrapped DQN additionally tracks:
- `update_counts`: Updates per head (should be balanced)

## Visualization

The `analyze_results.py` script creates:

### Learning Curves
- X-axis: Episodes
- Y-axis: Reward (100-episode moving average)
- Two lines: Vanilla (blue) vs Bootstrapped (purple)
- Saved to: `plots/vanilla_vs_bootstrapped_comparison.png`

### Performance Metrics
- Final reward comparison
- Episodes to convergence
- Sample efficiency ratios
- Timestep reduction percentage

## Analysis Functions

### In `prime_experiment.py`
```python
analyze_sample_efficiency()  # Compare convergence speed
plot_learning_curves()       # Generate comparison plots
```

### In `analyze_results.py`
```python
analyze_sample_efficiency()  # Detailed efficiency analysis
plot_learning_curves()       # Publication-quality plots
generate_comparison_table()  # Markdown table
```

## Interpretation Guide

### Good BDQN Performance Indicators

✅ **Faster initial learning**: BDQN curve rises faster in first 1000 episodes
✅ **Smoother curves**: Less variance due to uncertainty-driven exploration
✅ **30%+ sample efficiency gain**: Reaches thresholds in fewer episodes
✅ **Balanced head updates**: All heads get similar number of updates
✅ **Similar/better final performance**: At least matches vanilla DQN

### Warning Signs

⚠️ **Unbalanced heads**: Some heads get 10x more updates than others
⚠️ **Slower convergence**: Takes more episodes than vanilla (rare)
⚠️ **High variance**: Learning curve very noisy (suggests warmup too short)
⚠️ **Lower final performance**: Significantly worse than vanilla (check hyperparameters)

## Troubleshooting

### If BDQN is slower:
- Increase `warmup_steps` (try 10,000)
- Increase `init_q_range` (try 0.5)
- Check `mask_prob` is 0.8 (not too low)

### If unbalanced heads:
- Check replay buffer has enough data
- Verify per-step masking is working
- Increase `mask_prob` slightly (try 0.85)

### If poor exploration:
- Verify Thompson sampling is active
- Check `exploration_strategy='thompson'`
- Try UCB as alternative: `exploration_strategy='ucb'`

## Comparison with Four Rooms

| Aspect | Four Rooms (Tabular) | Boxman (Deep) |
|--------|---------------------|---------------|
| **State space** | ~100 states | ~∞ (images) |
| **Q-function** | Dictionary | Neural network |
| **Sample efficiency** | Direct updates | Gradient descent |
| **Convergence time** | Minutes | Hours/Days |
| **BDQN benefit** | 2-3x speedup | 1.3-1.5x speedup |

## Next Steps

1. **Run experiment**: `python prime_experiment.py`
2. **Analyze results**: `python analyze_results.py`
3. **Compare with four_rooms**: Check if sample efficiency gains are similar
4. **Try different tasks**: Add more tasks to `Tasks` list
5. **Boolean composition**: Test composed goals (AND, OR, NOT)

## Citation

This implementation combines:
- **Bootstrapped DQN**: Osband et al., "Deep Exploration via Bootstrapped DQN", NIPS 2016
- **Boolean Task Algebra**: Barreto et al., "A Boolean Task Algebra for Reinforcement Learning", 2020
- **DQN**: Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015

## Files Modified

1. ✅ `prime_experiment.py` - Main experiment script
2. ✅ `bdqn.py` - Bootstrapped DQN implementation (new)
3. ✅ `analyze_results.py` - Results analysis (new)
4. ✅ `BDQN_README.md` - BDQN documentation (new)
5. ✅ This file - Experiment documentation (new)
