import numpy as np
from matplotlib import pyplot as plt
import deepdish as dd
from GridWorld import GridWorld
from bdqn_library import (
    Bootstrapped_Goal_Oriented_Q_learning,
    AND, OR, NOT,
    EQ_P, EQ_V,
)

# ============================================================
#  Experiment setup
# ============================================================

T_states = [(3,3),(3,9),(9,3),(9,9)]
T_states = [[pos,pos] for pos in T_states]

Bases = [[(3,3),(3,9)], [(3,3),(9,3)]]
Tasks = [
    [],[(3,3),(3,9),(9,3),(9,9)],[(3,3)],[(3,9)],[(9,3)],[(9,9)],
    [(3,3),(3,9)],[(9,3),(9,9)],[(3,3),(9,3)],[(3,9),(9,9)],
    [(3,3),(3,9),(9,3)],[(3,3),(3,9),(9,9)],[(3,3),(9,3),(9,9)],
    [(3,9),(9,3),(9,9)],[(3,3),(9,9)],[(3,9),(9,3)]
]

# Load optimal solutions for convergence checking (same as exp1_bdqn.py)
Qs = dd.io.load('exps_data/4Goals_Optimal_Qs.h5')
Qs = [{s:v for (s,v) in Q} for Q in Qs]
EQs = dd.io.load('exps_data/4Goals_Optimal_EQs.h5')
EQs = [{s:{s__:v__ for (s__,v__) in v} for (s,v) in EQ} for EQ in EQs]

# (dense_reward?, same_terminal_states?)
types = [(True,True),(True,False),(False,True),(False,False)]

# Base BDQN parameters 
base_n_heads = 10        
base_mask_prob = 0.8     
base_alpha = 1          
base_init_q_range = 0.2  
base_warmup_steps = 30000  

# Policy agreement stopping condition
base_policy_agreement_threshold = 0.98  

# POLICY-ONLY Convergence Parameters
base_optimality_threshold = 0.3      
value_epsilon = 1.0               
evf_epsilon = 0.1               

# Environment-specific parameter adjustments for challenging scenarios
def get_bdqn_params(env_type):
    """Get BDQN parameters adjusted for specific environment challenges."""
    
    # Default parameters
    params = {
        'n_heads': base_n_heads,
        'mask_prob': base_mask_prob, 
        'alpha': base_alpha,
        'init_q_range': base_init_q_range,
        'warmup_steps': base_warmup_steps,
        'policy_agreement_threshold': base_policy_agreement_threshold,
        'optimality_threshold': base_optimality_threshold,
        'evf_epsilon': evf_epsilon
    }
    
    # Adjustments for challenging environments (types 1 and 2)
    if env_type in [1, 2]:  # Different absorbing set environments
        print(f"    Using enhanced parameters for challenging environment type {env_type}")
        params.update({
            'n_heads': 15,              # More ensemble diversity
            'mask_prob': 0.6,           # Lower masking for more stable learning
            'alpha': 0.8,               # Slightly lower learning rate
            'warmup_steps': 50000,      # Longer exploration phase
            'policy_agreement_threshold': 0.95,  # Slightly relaxed agreement
            'optimality_threshold': 0.25,        # More lenient optimality check
        })
        
        # Extra adjustments for sparse + different absorbing (type 1) - most challenging
        if env_type == 1:
            print(f"    Extra adjustments for sparse rewards + different absorbing set")
            params.update({
                'n_heads': 20,              # Maximum ensemble diversity
                'warmup_steps': 80000,      # Extended exploration
                'policy_agreement_threshold': 0.90,  # More relaxed for convergence
            })
    
    return params

# Experiment parameters
num_runs = 1000    # evaluation sample size

# ============================================================
#  Utility functions
# ============================================================

def get_optimal_eq_for_task(task_goals):
    """
    Find the optimal EVF Q-function for a given task.
    
    Args:
        task_goals: List of goal positions for the task (e.g., [(3,3), (9,9)])
    
    Returns:
        The corresponding optimal EVF Q-function from EQs, or None if not found
    """
    # Find the task index in the Tasks list
    try:
        task_idx = Tasks.index(task_goals)
        return EQs[task_idx]
    except (ValueError, IndexError):
        # Task not found in precomputed optimal solutions
        return None

def ensemble_to_evf(Q_input):
    """Combine bootstrapped EVFs using GPI-style max aggregation."""
    # If input is already a single EVF (not a list), return it directly
    if isinstance(Q_input, dict) and not isinstance(Q_input, list):
        return Q_input
    
    # Otherwise, it's a list of Q-functions to combine
    EQ_combined = {}
    for Q in Q_input:
        for s in Q:
            if s not in EQ_combined:
                EQ_combined[s] = {}
            for g in Q[s]:
                if g not in EQ_combined[s]:
                    EQ_combined[s][g] = Q[s][g].copy()
                else:
                    EQ_combined[s][g] = np.maximum(EQ_combined[s][g], Q[s][g])
    return EQ_combined


def single_evaluate(goals, EQ):
    """Evaluate a single EVF with deterministic policy."""
    env = GridWorld(goals=goals, T_states=T_states)
    policy = EQ_P(EQ)
    state = env.reset()
    done = False
    t = 0
    G = 0
    while not done and t < 100:
        action = policy[state] if state in policy else np.random.randint(env.action_space.n)
        state_, reward, done, _ = env.step(action)
        state = state_
        G += reward
        t += 1
    return G

def is_wall(row, col):
    """Define wall positions based on the Four Rooms MAP"""
    # Define the exact wall pattern from the Four Rooms environment
    map_walls = [
        # Row 0: all walls
        *[(0, j) for j in range(13)],
        # Row 6: horizontal walls with openings
        (6, 1), (6, 3), (6, 4), (6, 5), (6, 6),
        # Row 7: partial horizontal walls  
        (7, 0), (7, 6), (7, 7), (7, 8), (7, 9), (7, 11), (7, 12),
        # Row 12: all walls
        *[(12, j) for j in range(13)],
        # Vertical walls at column 0 and 12
        *[(i, 0) for i in range(13)],
        *[(i, 12) for i in range(13)],
        # Vertical wall at column 6 with openings at rows 2 and 10
        (1, 6), (3, 6), (4, 6), (5, 6), (7, 6), (8, 6), (9, 6), (11, 6)
    ]
    
    return (row, col) in map_walls


# ============================================================
#  Main experiment loop
# ============================================================

for t in range(len(types)):
    print(f"=== Type {t} ({types[t]}) ===")
    
    # Get environment-specific parameters
    params = get_bdqn_params(t)
    
    # -------------------------------
    # Learn EQ_max and EQ_min bounds
    # -------------------------------
    print("  Learning EQ_max ...")
    env = GridWorld(goals=T_states, dense_rewards=not types[t][0])
    # Find optimal solution for all 4 goals task (index 1 in Tasks list)
    eq_max_optimal = get_optimal_eq_for_task([(3,3),(3,9),(9,3),(9,9)])
    EQ_max, _ = Bootstrapped_Goal_Oriented_Q_learning(
        env, T_states=T_states, Q_optimal=eq_max_optimal, 
        mask_prob=params['mask_prob'],
        n_heads=params['n_heads'], 
        alpha=params['alpha'], 
        init_q_range=params['init_q_range'],
        warmup_steps=params['warmup_steps'], 
        policy_agreement_threshold=params['policy_agreement_threshold'],
        optimality_threshold=params['optimality_threshold'], 
        evf_epsilon=params['evf_epsilon']
    )

    print("  Learning EQ_min ...")
    env = GridWorld(goals=T_states, goal_reward=-0.1, dense_rewards=not types[t][0])
    # EQ_min uses different reward structure, so no precomputed optimal available
    EQ_min, _ = Bootstrapped_Goal_Oriented_Q_learning(
        env, T_states=T_states, Q_optimal=None, 
        mask_prob=params['mask_prob'],
        n_heads=params['n_heads'], 
        alpha=params['alpha'], 
        init_q_range=params['init_q_range'],
        warmup_steps=params['warmup_steps'], 
        policy_agreement_threshold=params['policy_agreement_threshold'],
        optimality_threshold=params['optimality_threshold'], 
        evf_epsilon=params['evf_epsilon']
    )

    # -------------------------------
    # Learn base tasks A and B
    # -------------------------------
    print("  Learning base task A ...")
    goals = [[pos,pos] for pos in Bases[0]]
    env = GridWorld(goals=goals, dense_rewards=not types[t][0],
                    T_states=T_states if types[t][1] else goals)
    # Base task A is [(3,3),(3,9)] - find optimal solution
    a_optimal = get_optimal_eq_for_task(Bases[0])
    A, _ = Bootstrapped_Goal_Oriented_Q_learning(
        env, T_states=None if types[t][1] else T_states, Q_optimal=a_optimal,
        mask_prob=params['mask_prob'],
        n_heads=params['n_heads'], 
        alpha=params['alpha'], 
        init_q_range=params['init_q_range'],
        warmup_steps=params['warmup_steps'], 
        policy_agreement_threshold=params['policy_agreement_threshold'],
        optimality_threshold=params['optimality_threshold'], 
        evf_epsilon=params['evf_epsilon']
    )

    print("  Learning base task B ...")
    goals = [[pos,pos] for pos in Bases[1]]
    env = GridWorld(goals=goals, dense_rewards=not types[t][0],
                    T_states=T_states if types[t][1] else goals)
    # Base task B is [(3,3),(9,3)] - find optimal solution
    b_optimal = get_optimal_eq_for_task(Bases[1])
    B, _ = Bootstrapped_Goal_Oriented_Q_learning(
        env, T_states=None if types[t][1] else T_states, Q_optimal=b_optimal,
        mask_prob=params['mask_prob'],
        n_heads=params['n_heads'], 
        alpha=params['alpha'], 
        init_q_range=params['init_q_range'],
        warmup_steps=params['warmup_steps'], 
        policy_agreement_threshold=params['policy_agreement_threshold'],
        optimality_threshold=params['optimality_threshold'], 
        evf_epsilon=params['evf_epsilon']
    )

    # -------------------------------
    # Compose new tasks using logic
    # -------------------------------
    NEG = lambda x: NOT(x, EQ_max=EQ_max, EQ_min=EQ_min)
    XOR = lambda EQ1, EQ2: OR(AND(EQ1, NEG(EQ2)), AND(EQ2, NEG(EQ1)))

    composed = [
        EQ_min,                              # []
        EQ_max,                              # [(3,3),(3,9),(9,3),(9,9)]
        AND(A,B),                            # [(3,3)]
        AND(A,NEG(B)),                       # [(3,9)]
        AND(B,NEG(A)),                       # [(9,3)]
        NEG(OR(A,B)),                        # [(9,9)]
        A,                                   # [(3,3),(3,9)]
        NEG(A),                              # [(9,3),(9,9)]
        B,                                   # [(3,3),(9,3)]
        NEG(B),                              # [(3,9),(9,9)]
        OR(A,B),                             # [(3,3),(3,9),(9,3)]
        OR(A,NEG(B)),                        # [(3,3),(3,9),(9,9)]
        OR(B,NEG(A)),                        # [(3,3),(9,3),(9,9)]
        NEG(AND(A,B)),                       # [(3,9),(9,3),(9,9)]
        NEG(XOR(A,B)),                       # [(3,3),(9,9)]
        XOR(A,B)                             # [(3,9),(9,3)]
    ]

    # -------------------------------
    # Save action policies as matrices
    # -------------------------------
    print("  Saving action policy matrices...")
    
    # Create a 13x13x16 matrix to store actions for each task
    # 13x13 for Four Rooms grid, 16 for number of tasks
    action_matrices = np.full((13, 13, len(Tasks)), -1, dtype=int)  # -1 for walls/invalid states
    
    for task_idx, task_goals in enumerate(Tasks):
        # Extract policy for this task
        EQ = composed[task_idx]
        policy = EQ_P(EQ)
        
        # Fill the 13x13 grid with actions
        for row in range(13):
            for col in range(13):
                state = (row, col)
                if state in policy:
                    action_matrices[row, col, task_idx] = policy[state]
                elif is_wall(row, col):
                    action_matrices[row, col, task_idx] = -1  # Wall marker
                else:
                    action_matrices[row, col, task_idx] = -2  # Unknown/unreachable state
    
    # Save action matrices
    action_fname = f"exps_data/exp3_bdqn_actions_{t}.h5"
    dd.io.save(action_fname, action_matrices)
    print(f"  Action matrices saved to {action_fname}")
    
    # Also save a readable mapping of task indices to task names
    task_info = {
        'task_goals': Tasks,
        'task_names': [
            'Empty', 'All Goals', 'MT AND ML', 'MT AND NOT ML', 'ML AND NOT MT', 'NOT MT AND NOT ML',
            'MT', 'NOT MT', 'ML', 'NOT ML', 'MT OR ML', 'MT OR NOT ML', 'ML OR NOT MT', 
            'NOT MT OR NOT ML', 'NOT (MT XOR ML)', 'MT XOR ML'
        ],
        'action_encoding': {
            '-1': 'Wall',
            '-2': 'Unknown/Unreachable', 
            '0': 'Up',
            '1': 'Right', 
            '2': 'Down',
            '3': 'Left',
            '4': 'Stay'
        }
    }
    
    task_info_fname = f"exps_data/exp3_bdqn_task_info_{t}.h5"
    dd.io.save(task_info_fname, task_info)
    print(f"  Task info saved to {task_info_fname}")

    # -------------------------------
    # Evaluate composed tasks
    # -------------------------------
    print("  Starting evaluation phase...")
    data = np.zeros((num_runs, len(Tasks)))
    for i in range(num_runs):
        if i % 1000 == 0:
            print(f"    Evaluation run {i}/{num_runs}")
        for j in range(len(Tasks)):
            goals = [[pos,pos] for pos in Tasks[j]]
            data[i, j] = single_evaluate(goals, composed[j])

    fname = f"exps_data/exp3_bdqn_returns_{t}.h5"
    dd.io.save(fname, data)
    print(f"  Completed type {t}, results saved to {fname}")
    print(f"  Average returns: {np.mean(data, axis=0)}")
    print()

print("=== Experiment finished successfully ===")
