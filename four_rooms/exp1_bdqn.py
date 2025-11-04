import numpy as np
from matplotlib import pyplot as plt
import deepdish as dd
from GridWorld import GridWorld
from bdqn_library import *

# Exact replication of exp1.py but using Bootstrapped DQN instead of epsilon-greedy

env = GridWorld()
maxiter = 3000  # Increased for Bootstrapped DQN convergence
T_states = [(3,3),(3,9),(9,3),(9,9)]
T_states = [[pos,pos] for pos in T_states]
Tasks = [
    [],[(3,3),(3,9),(9,3),(9,9)],[(3,3)],[(3,9)],[(9,3)],[(9,9)],
    [(3,3),(3,9)],[(3,9),(9,3)],[(9,3),(9,9)],[(3,3),(9,3)],
    [(3,3),(3,9),(9,3)],[(3,3),(3,9),(9,9)],[(3,3),(9,3),(9,9)],
    [(3,9),(9,3),(9,9)],[(3,3),(9,9)],[(3,9),(9,3)]
]

# Load optimal solutions for convergence checking (same as exp1.py)
Qs = dd.io.load('exps_data/4Goals_Optimal_Qs.h5')
Qs = [{s:v for (s,v) in Q} for Q in Qs]
EQs = dd.io.load('exps_data/4Goals_Optimal_EQs.h5')
EQs = [{s:{s__:v__ for (s__,v__) in v} for (s,v) in EQ} for EQ in EQs]

num_runs = 1
dataQ = np.zeros((num_runs,len(Tasks))) 
dataEQ = np.zeros((num_runs,len(Tasks))) 

# Bootstrapped DQN parameters optimized for per-transaction masking
n_heads = 10        # Moderate ensemble size for good diversity vs convergence
mask_prob = 0.8     # Per-transaction masking: each transition included in ~80% of heads
alpha_q = 1       # Learning rate for Q-learning
alpha_eq = 1      # Learning rate for goal-oriented Q-learning
init_q_range = 0.2  # Positive random Q-value initialization range [0, init_q_range]
warmup_steps = 500  # Warmup phase for better per-transaction diversity

# Policy agreement stopping condition
policy_agreement_threshold = 0.95  # 95% of states must have policy agreement

# POLICY-ONLY Convergence Parameters (ignores value differences, focuses on behavior)
optimality_threshold = 0.8      # 80% of states must match optimal policy
value_epsilon = 1.0             # [UNUSED in policy-only mode] 
evf_epsilon = 0.1               # [UNUSED in policy-only mode]

idxs = np.arange(len(Tasks))
for i in range(num_runs):
    print("run: ",i)
    np.random.shuffle(idxs)
    for j in idxs:
        print(f"Task: {j} (goals: {Tasks[j]})")
        goals = [[pos,pos] for pos in Tasks[j]]
        env = GridWorld(goals=goals, goal_reward=1, step_reward=-0.01, T_states=T_states)
        
        # Bootstrapped Q-learning with ensemble policy agreement stopping condition
        print(f"  Starting Q-learning...")
        _, stats = Bootstrapped_Q_learning(env, Q_optimal=Qs[j], maxiter=maxiter, 
                                         n_heads=n_heads, mask_prob=mask_prob, alpha=alpha_q,
                                         init_q_range=init_q_range, warmup_steps=warmup_steps,
                                         policy_agreement_threshold=policy_agreement_threshold,
                                         optimality_threshold=optimality_threshold,
                                         value_epsilon=value_epsilon)
        dataQ[i,j] = stats["T"]
        print(f"  Q-learning completed in {stats['T']} steps")
        
        # Bootstrapped Goal-Oriented Q-learning with ensemble policy agreement stopping condition
        print(f"  Starting Goal-Oriented Q-learning...")
        _, stats = Bootstrapped_Goal_Oriented_Q_learning(env, T_states=T_states, Q_optimal=EQs[j], 
                                                        maxiter=maxiter, n_heads=n_heads, 
                                                        mask_prob=mask_prob, alpha=alpha_eq,
                                                        init_q_range=init_q_range, warmup_steps=warmup_steps,
                                                        policy_agreement_threshold=policy_agreement_threshold,
                                                        optimality_threshold=optimality_threshold,
                                                        evf_epsilon=evf_epsilon)
        dataEQ[i,j] = stats["T"]
        print(f"  Goal-Oriented Q-learning completed in {stats['T']} steps")

# Save results with bdqn suffix to distinguish from original exp1
data1 = dd.io.save('exps_data/exp1_bdqn_samples_Qs.h5', dataQ)
data2 = dd.io.save('exps_data/exp1_bdqn_samples_EQs.h5', dataEQ)

# Print summary statistics for comparison
print(f"\n=== BDQN vs Original Comparison ===")
print(f"Standard Q-learning - BDQN avg samples: {dataQ.mean():.1f} ± {dataQ.std():.1f}")
print(f"Goal-Oriented Q-learning - BDQN avg samples: {dataEQ.mean():.1f} ± {dataEQ.std():.1f}")
print(f"Results saved to exp1_bdqn_samples_*.h5")
print(f"Compare with original exp1_samples_*.h5 files for sample efficiency analysis")