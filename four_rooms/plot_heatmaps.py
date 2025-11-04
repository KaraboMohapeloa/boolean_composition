#!/usr/bin/env python3
"""
Heatmap visualization for boolean composition task performance analysis.
Uses MT and ML notation, excludes negation tasks, saves plots without displaying.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import deepdish as dd
import pandas as pd
from bdqn_library import AND, OR, NOT, EQ_P
from GridWorld import GridWorld

def plot_majority_vote_policies():
    """
    Generate policy visualization heatmaps showing the BDQN policies
    extracted from optimal Q-functions for MT/ML tasks across all 4 environment scenarios.
    """
    # Task names and indices
    task_names = ['MT AND ML', 'MT', 'ML', 'MT OR ML']
    task_indices = [2, 6, 8, 10]
    
    # Environment scenarios
    env_scenarios = [
        "Dense rewards, Same absorbing set",      # type 0
        "Sparse rewards, Different absorbing set", # type 1  
        "Dense rewards, Different absorbing set",  # type 2
        "Sparse rewards, Same absorbing set"       # type 3
    ]
    
    # Action directions: 0=up, 1=right, 2=down, 3=left, 4=stay
    action_arrows = {0: '↑', 1: '→', 2: '↓', 3: '←', 4: '●'}
    
    # Generate plot only for sparse rewards, same absorbing set (type 3)
    env_type = 3
    env_name = env_scenarios[env_type]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    print(f"Extracting BDQN policies for: {env_name}")
    
    for task_idx, (task_name, data_idx) in enumerate(zip(task_names, task_indices)):
        ax = axes[task_idx]
        
        # Create a 13x13 grid for the four rooms environment
        grid_size = 13
        policy_grid = np.zeros((grid_size, grid_size), dtype=object)
        
        # Extract policy from optimal Q-functions
        learned_policy = extract_bdqn_policy(env_type, data_idx)
        
        # Fill the policy grid with extracted policy
        for i in range(grid_size):
            for j in range(grid_size):
                if is_wall(i, j):
                    policy_grid[i, j] = '█'  # Wall
                elif is_goal(i, j, data_idx):
                    policy_grid[i, j] = '★'  # Goal
                else:
                    # Use extracted policy if available
                    if learned_policy is not None and (i, j) in learned_policy:
                        optimal_action = learned_policy[(i, j)]
                        policy_grid[i, j] = action_arrows.get(optimal_action, '?')
                    else:
                        policy_grid[i, j] = '?'  # No policy available
        
        # Create the visualization
        create_policy_heatmap(ax, policy_grid, f"{task_name}")
    
    # Add overall title
    fig.suptitle(f'BDQN Policy from Optimal Q-functions: {env_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save with environment-specific filename
    env_filename = env_name.lower().replace(' ', '_').replace(',', '')
    plt.savefig(f"plots/bdqn_policy_{env_filename}.pdf", bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  Saved: plots/bdqn_policy_{env_filename}.pdf")

def plot_policy_arrows():
    """
    Generate policy visualization heatmaps showing the actual learned BDQN policies
    for MT/ML tasks across all 4 environment scenarios to evaluate if BDQN learned correctly.
    """
    # Task names and indices
    task_names = ['MT AND ML', 'MT', 'ML', 'MT OR ML']
    task_indices = [2, 6, 8, 10]
    
    # Environment scenarios
    env_scenarios = [
        "Dense rewards, Same absorbing set",      # type 0
        "Sparse rewards, Different absorbing set", # type 1  
        "Dense rewards, Different absorbing set",  # type 2
        "Sparse rewards, Same absorbing set"       # type 3
    ]
    
    # Action directions: 0=up, 1=right, 2=down, 3=left, 4=stay
    action_arrows = {0: '↑', 1: '→', 2: '↓', 3: '←', 4: '●'}
    
    # Generate separate plot for each environment scenario
    for env_type, env_name in enumerate(env_scenarios):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        print(f"Extracting BDQN learned policy for: {env_name}")
        
        for task_idx, (task_name, data_idx) in enumerate(zip(task_names, task_indices)):
            ax = axes[task_idx]
            
            # Create a 13x13 grid for the four rooms environment
            grid_size = 13
            policy_grid = np.zeros((grid_size, grid_size), dtype=object)
            
            # Try to extract actual learned BDQN policy
            learned_policy = extract_bdqn_policy(env_type, data_idx)
            
            # Fill the policy grid with learned or fallback policy
            for i in range(grid_size):
                for j in range(grid_size):
                    if is_wall(i, j):
                        policy_grid[i, j] = '█'  # Wall
                    elif is_goal(i, j, data_idx):
                        policy_grid[i, j] = '★'  # Goal
                    else:
                        # Use learned policy if available, otherwise show placeholder
                        if learned_policy is not None:
                            try:
                                # Try to get action for this position
                                if (i, j) in learned_policy:
                                    optimal_action = learned_policy[(i, j)]
                                    policy_grid[i, j] = action_arrows[optimal_action]
                                else:
                                    # Position not in policy - use fallback
                                    policy_grid[i, j] = '?'  # Unknown/not learned
                            except Exception as e:
                                # Fallback if action extraction fails
                                policy_grid[i, j] = '?'  # Failed to extract
                        else:
                            policy_grid[i, j] = '?'  # No learned policy available
            
            # Create the visualization
            create_policy_heatmap(ax, policy_grid, f"{task_name}")
        
        # Add overall title indicating this shows learned BDQN policy
        fig.suptitle(f'Learned BDQN Policy: {env_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save with environment-specific filename
        env_filename = env_name.lower().replace(' ', '_').replace(',', '')
        plt.savefig(f"plots/bdqn_learned_policy_{env_filename}.pdf", bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  Saved: plots/bdqn_learned_policy_{env_filename}.pdf")

def extract_bdqn_policy(env_type, task_idx):
    """
    Extract BDQN policy by recreating the exact same learning process as exp3_bdqn.py.
    Returns the policy as a mapping from (row, col) to optimal action.
    """
    try:
        from bdqn_library import Bootstrapped_Goal_Oriented_Q_learning
        
        # Recreate the same setup as exp3_bdqn.py
        T_states = [(3,3),(3,9),(9,3),(9,9)]
        T_states = [[pos,pos] for pos in T_states]
        Bases = [[(3,3),(3,9)], [(3,3),(9,3)]]
        Tasks = [
            [],[(3,3),(3,9),(9,3),(9,9)],[(3,3)],[(3,9)],[(9,3)],[(9,9)],
            [(3,3),(3,9)],[(9,3),(9,9)],[(3,3),(9,3)],[(3,9),(9,9)],
            [(3,3),(3,9),(9,3)],[(3,3),(3,9),(9,9)],[(3,3),(9,3),(9,9)],
            [(3,9),(9,3),(9,9)],[(3,3),(9,9)],[(3,9),(9,3)]
        ]
        types = [(True,True),(True,False),(False,True),(False,False)]
        
        # Load optimal solutions for convergence checking
        try:
            EQs = dd.io.load('exps_data/4Goals_Optimal_EQs.h5')
            EQs = [{s:{s__:v__ for (s__,v__) in v} for (s,v) in EQ} for EQ in EQs]
        except:
            print(f"    Warning: Could not load optimal solutions")
            EQs = [None] * 16
        
        # Use the same BDQN parameters as exp3_bdqn.py
        n_heads = 10
        mask_prob = 0.9
        alpha = 1
        init_q_range = 0.2
        warmup_steps = 20000
        policy_agreement_threshold = 0.96
        optimality_threshold = 0.3
        evf_epsilon = 0.1
        
        # Learn the same base components as exp3_bdqn.py
        print(f"    Learning EQ_max for env_type {env_type}...")
        env = GridWorld(goals=T_states, dense_rewards=not types[env_type][0])
        eq_max_optimal = EQs[1] if EQs[1] is not None else None
        EQ_max, _ = Bootstrapped_Goal_Oriented_Q_learning(
            env, T_states=T_states, Q_optimal=eq_max_optimal, mask_prob=mask_prob,
            n_heads=n_heads, alpha=alpha, init_q_range=init_q_range,
            warmup_steps=warmup_steps, policy_agreement_threshold=policy_agreement_threshold,
            optimality_threshold=optimality_threshold, evf_epsilon=evf_epsilon
        )
        
        print(f"    Learning EQ_min for env_type {env_type}...")
        env = GridWorld(goals=T_states, goal_reward=-0.1, dense_rewards=not types[env_type][0])
        EQ_min, _ = Bootstrapped_Goal_Oriented_Q_learning(
            env, T_states=T_states, Q_optimal=None, mask_prob=mask_prob,
            n_heads=n_heads, alpha=alpha, init_q_range=init_q_range,
            warmup_steps=warmup_steps, policy_agreement_threshold=policy_agreement_threshold,
            optimality_threshold=optimality_threshold, evf_epsilon=evf_epsilon
        )
        
        print(f"    Learning A for env_type {env_type}...")
        goals = [[pos,pos] for pos in Bases[0]]
        env = GridWorld(goals=goals, dense_rewards=not types[env_type][0],
                        T_states=T_states if types[env_type][1] else goals)
        a_optimal = EQs[6] if EQs[6] is not None else None
        A, _ = Bootstrapped_Goal_Oriented_Q_learning(
            env, T_states=None if types[env_type][1] else T_states, Q_optimal=a_optimal,
            mask_prob=mask_prob, n_heads=n_heads, alpha=alpha, init_q_range=init_q_range,
            warmup_steps=warmup_steps, policy_agreement_threshold=policy_agreement_threshold,
            optimality_threshold=optimality_threshold, evf_epsilon=evf_epsilon
        )
        
        print(f"    Learning B for env_type {env_type}...")
        goals = [[pos,pos] for pos in Bases[1]]
        env = GridWorld(goals=goals, dense_rewards=not types[env_type][0],
                        T_states=T_states if types[env_type][1] else goals)
        b_optimal = EQs[8] if EQs[8] is not None else None
        B, _ = Bootstrapped_Goal_Oriented_Q_learning(
            env, T_states=None if types[env_type][1] else T_states, Q_optimal=b_optimal,
            mask_prob=mask_prob, n_heads=n_heads, alpha=alpha, init_q_range=init_q_range,
            warmup_steps=warmup_steps, policy_agreement_threshold=policy_agreement_threshold,
            optimality_threshold=optimality_threshold, evf_epsilon=evf_epsilon
        )
        
        # Create the composed tasks exactly as exp3_bdqn.py does
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
        
        # Extract policy for the requested task
        if task_idx < len(composed):
            EQ = composed[task_idx]
            policy = EQ_P(EQ)
            print(f"    Task {task_idx}: Extracted policy for {len(policy)} states")
            return policy
        else:
            print(f"    Task {task_idx} out of range")
            return None
        
    except Exception as e:
        print(f"    Error extracting policy for task {task_idx}, env {env_type}: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_heuristic_action(row, col, task_idx):
    """
    Generate a reasonable heuristic action for a given state and task.
    Action encoding: 0=up, 1=right, 2=down, 3=left
    """
    # Define goal positions for MT and ML
    # MT (top-left room): around (2, 2)
    # ML (bottom-right room): around (10, 10)
    
    mt_goal = (2, 2)
    ml_goal = (10, 10)
    
    # Task-specific heuristics
    if task_idx == 2:  # MT AND ML - go to whichever is closer first
        dist_to_mt = abs(row - mt_goal[0]) + abs(col - mt_goal[1])
        dist_to_ml = abs(row - ml_goal[0]) + abs(col - ml_goal[1])
        target = mt_goal if dist_to_mt <= dist_to_ml else ml_goal
    elif task_idx == 6:  # MT only
        target = mt_goal
    elif task_idx == 8:  # ML only  
        target = ml_goal
    elif task_idx == 10:  # MT OR ML - go to whichever is closer
        dist_to_mt = abs(row - mt_goal[0]) + abs(col - mt_goal[1])
        dist_to_ml = abs(row - ml_goal[0]) + abs(col - ml_goal[1])
        target = mt_goal if dist_to_mt <= dist_to_ml else ml_goal
    else:
        # Default behavior
        target = mt_goal
    
    # Calculate direction to target
    row_diff = target[0] - row
    col_diff = target[1] - col
    
    # Choose action based on largest difference (Manhattan distance heuristic)
    if abs(row_diff) > abs(col_diff):
        if row_diff > 0:
            return 2  # down
        else:
            return 0  # up
    else:
        if col_diff > 0:
            return 1  # right
        else:
            return 3  # left

def position_to_state_index(row, col):
    """Convert grid position to state index used in your data structure"""
    # This depends on how your environment indexes states
    # Four rooms typically uses flattened indexing
    return row * 13 + col

def is_wall(row, col):
    """Define wall positions based on the Four Rooms MAP"""
    # Define the exact wall pattern from the MAP
    walls = [
        # Top and bottom borders
        (0, slice(None)), (12, slice(None)),
        # Left and right borders  
        (slice(None), 0), (slice(None), 12),
        # Internal walls
        # Vertical wall at col=6, with openings at (2,6) and (10,6)
        (1, 6), (3, 6), (4, 6), (5, 6), (7, 6), (8, 6), (9, 6), (11, 6),
        # Horizontal walls
        (6, 1), (6, 3), (6, 4), (6, 5), (6, 6),  # Left part of horizontal wall
        (7, 7), (7, 8), (7, 9), (7, 11)  # Right part of horizontal wall
    ]
    
    # Check against the exact MAP pattern
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

def is_goal(row, col, task_idx):
    """Define goal positions for each task based on the Four Rooms corners"""
    goals = {
        2: [(3, 3)],                    # MT AND ML - intersection (top-left room)
        6: [(3, 3), (3, 9)],           # MT - top row goals
        8: [(3, 3), (9, 3)],           # ML - left column goals  
        10: [(3, 3), (3, 9), (9, 3)]   # MT OR ML - union of both
    }
    return (row, col) in goals.get(task_idx, [])

def get_direction_to_nearest_goal(row, col, task_idx):
    """Get the direction that moves toward the nearest goal"""
    goals = {
        2: [(3, 3)],                    
        6: [(3, 3), (3, 9)],           
        8: [(3, 3), (9, 3)],           
        10: [(3, 3), (3, 9), (9, 3)]   
    }
    
    task_goals = goals.get(task_idx, [(3, 3)])
    
    # Find nearest goal
    min_dist = float('inf')
    nearest_goal = task_goals[0]
    
    for goal in task_goals:
        dist = abs(row - goal[0]) + abs(col - goal[1])  # Manhattan distance
        if dist < min_dist:
            min_dist = dist
            nearest_goal = goal
    
    goal_row, goal_col = nearest_goal
    
    # Determine direction to move toward goal
    row_diff = goal_row - row
    col_diff = goal_col - col
    
    # Prioritize the larger difference
    if abs(row_diff) > abs(col_diff):
        if row_diff > 0:
            return 2  # DOWN
        else:
            return 0  # UP
    else:
        if col_diff > 0:
            return 1  # RIGHT
        else:
            return 3  # LEFT

def create_policy_heatmap(ax, policy_grid, title):
    """Create a heatmap visualization with arrows"""
    grid_size = policy_grid.shape[0]
    
    # Create background grid with different colors for walls, goals, and regular states
    background = np.zeros((grid_size, grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            if policy_grid[i, j] == '█':  # Wall
                background[i, j] = -1
            elif policy_grid[i, j] == '★':  # Goal
                background[i, j] = 1
            else:  # Regular state
                background[i, j] = 0
    
    # Create heatmap with Four Rooms colors
    colors_map = ['lightblue', 'lightgray', 'lightgreen']  # regular, goal, wall
    cmap = plt.cm.colors.ListedColormap(['darkgray', 'wheat', 'gold'])
    
    im = ax.imshow(background, cmap=cmap, vmin=-1, vmax=1)
    
    # Add arrows/symbols as text
    for i in range(grid_size):
        for j in range(grid_size):
            symbol = policy_grid[i, j]
            if symbol == '█':
                color = 'white'
                fontsize = 8
                fontweight = 'normal'
            elif symbol == '★':
                color = 'red'
                fontsize = 16
                fontweight = 'bold'
            else:
                # Use red and orange colors for arrows, and gray for unknown
                if symbol == '?':
                    color = 'gray'
                    fontsize = 10
                    fontweight = 'normal'
                elif symbol in ['↑', '↓']:
                    color = 'red'
                    fontsize = 12
                    fontweight = 'bold'
                else:  # '→', '←', '●'
                    color = 'darkorange'
                    fontsize = 12
                    fontweight = 'bold'
            
            ax.text(j, i, symbol, ha='center', va='center', 
                   color=color, fontsize=fontsize, fontweight=fontweight)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add subtle grid lines
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.2, alpha=0.5)

def load_policy_from_qvalues(task_idx, env_type=0):
    """
    Load Q-values and extract policy (placeholder function)
    You'll need to implement this based on your data structure
    """
    try:
        # Load your Q-value data
        # q_values = dd.io.load(f'path_to_q_values_{task_idx}_{env_type}.h5')
        
        # Extract policy by taking argmax over actions
        # policy = np.argmax(q_values, axis=-1)  # Assuming last dimension is actions
        
        # For now, return random policy as placeholder
        grid_size = 13
        policy = np.random.randint(0, 4, size=(grid_size, grid_size))
        return policy
        
    except:
        # Return default policy if data not available
        grid_size = 13
        return np.ones((grid_size, grid_size), dtype=int)  # All right arrows

def plot_conjunction_disjunction_heatmap():
    """
    Generate heatmap showing performance of MT/ML tasks only.
    Excludes negation tasks, saves without showing.
    """
    # Task names and indices (only MT, ML, and MT OR ML)
    task_names = [
        'MT AND ML',         # index 2: [(3,3)] - AND(A,B)
        'MT',                # index 6: [(3,3),(3,9)] - A
        'ML',                # index 8: [(3,3),(9,3)] - B  
        'MT OR ML'           # index 10: [(3,3),(3,9),(9,3)] - OR(A,B)
    ]
    
    task_indices = [2, 6, 8, 10]
    
    # Environment type names
    env_types = [
        'Dense rewards,\nSame absorbing set',        # type 0: (True,True)
        'Sparse rewards,\nDifferent absorbing set',   # type 1: (True,False) 
        'Dense rewards,\nDifferent absorbing set',    # type 2: (False,True)
        'Sparse rewards,\nSame absorbing set'         # type 3: (False,False)
    ]
    
    # Load data for all environment types
    heatmap_data = []
    for env_type in range(4):
        try:
            data = dd.io.load(f'exps_data/exp3_bdqn_returns_{env_type}.h5')
            # Extract mean performance for selected tasks
            task_means = [np.mean(data[:, idx]) for idx in task_indices]
            heatmap_data.append(task_means)
        except FileNotFoundError:
            print(f"Warning: exp3_bdqn_returns_{env_type}.h5 not found")
            # Fill with zeros if file doesn't exist
            heatmap_data.append([0.0] * len(task_indices))
    
    # Convert to numpy array for heatmap
    heatmap_data = np.array(heatmap_data)
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    rc_ = {'axes.labelsize': 16, 'xtick.labelsize': 14, 
           'ytick.labelsize': 14, 'font.size': 14}
    sns.set(rc=rc_, style="whitegrid")
    
    # Create heatmap
    ax = sns.heatmap(heatmap_data, 
                     xticklabels=task_names,
                     yticklabels=env_types,
                     annot=True, 
                     fmt='.3f',
                     cmap='RdYlBu_r',  # Red-Yellow-Blue colormap (reverse)
                     center=0,        # Center colormap at 0
                     cbar_kws={'label': 'Average Return'})
    
    plt.title('Boolean Task Performance: MT and ML\n(No Negation Tasks)', 
              fontsize=18, pad=20)
    plt.xlabel('Boolean Tasks', fontsize=16)
    plt.ylabel('Environment Configuration', fontsize=16)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0, ha='center')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig("plots/mt_ml_heatmap.pdf", bbox_inches='tight', dpi=300)
    plt.close()  # Close instead of show
    
    # Print summary statistics
    print("\n=== MT/ML Task Performance Summary ===")
    conjunction_perf = np.mean(heatmap_data[:, 0])  # MT AND ML
    mt_perf = np.mean(heatmap_data[:, 1])           # MT
    ml_perf = np.mean(heatmap_data[:, 2])           # ML
    disjunction_perf = np.mean(heatmap_data[:, 3])  # MT OR ML
    
    print(f"MT AND ML Performance: {conjunction_perf:.3f}")
    print(f"MT Performance: {mt_perf:.3f}")
    print(f"ML Performance: {ml_perf:.3f}")
    print(f"MT OR ML Performance: {disjunction_perf:.3f}")
    
    # Environment-wise analysis
    print("\nEnvironment-wise performance:")
    for i, env_name in enumerate(env_types):
        env_mean = np.mean(heatmap_data[i, :])
        print(f"{env_name.replace(chr(10), ' ')}: {env_mean:.3f}")


def plot_conjunction_disjunction_detailed():
    """
    Create detailed heatmap for MT/ML tasks with goal position mapping.
    No negation tasks, uses MT/ML notation.
    """
    # Goal positions for reference
    goal_positions = {
        'MT AND ML': '(3,3)',                    # MT∩ML - single goal
        'MT': '(3,3), (3,9)',                  # MT - two goals
        'ML': '(3,3), (9,3)',                  # ML - two goals  
        'MT OR ML': '(3,3), (3,9), (9,3)'       # MT∪ML - three goals
    }
    
    task_names = ['MT AND ML', 'MT', 'ML', 'MT OR ML']
    task_indices = [2, 6, 8, 10]
    
    env_types = [
        'Dense rewards,\nSame absorbing set',
        'Sparse rewards,\nDifferent absorbing set', 
        'Dense rewards,\nDifferent absorbing set',
        'Sparse rewards,\nSame absorbing set'
    ]
    
    # Load and prepare data
    task_data = []
    
    for env_type in range(4):
        try:
            data = dd.io.load(f'exps_data/exp3_bdqn_returns_{env_type}.h5')
            task_means = [np.mean(data[:, idx]) for idx in task_indices]
            task_data.append(task_means)
        except FileNotFoundError:
            print(f"Warning: exp3_bdqn_returns_{env_type}.h5 not found")
            task_data.append([0.0] * 4)
    
    task_data = np.array(task_data)
    
    # Create single detailed heatmap
    plt.figure(figsize=(12, 8))
    rc_ = {'axes.labelsize': 14, 'xtick.labelsize': 12, 
           'ytick.labelsize': 12, 'font.size': 12}
    sns.set(rc=rc_, style="whitegrid")
    
    # Create heatmap with goal positions
    ax = sns.heatmap(task_data, 
                     xticklabels=[f"{task}\n→ {goal_positions[task]}" for task in task_names],
                     yticklabels=env_types,
                     annot=True, 
                     fmt='.3f',
                     cmap='RdYlBu_r',
                     center=0,
                     cbar_kws={'label': 'Average Return'})
    
    ax.set_title('MT and ML Task Performance with Goal Positions', fontsize=16, pad=15)
    ax.set_xlabel('Task → Goal Positions', fontsize=14)
    ax.set_ylabel('Environment Configuration', fontsize=14)
    
    plt.tight_layout()
    plt.savefig("plots/mt_ml_detailed.pdf", bbox_inches='tight', dpi=300)
    plt.close()  # Close instead of show


def plot_task_correlation_matrix():
    """
    Create correlation matrix for MT/ML tasks only (no negation).
    """
    task_names = ['MT AND ML', 'MT', 'ML', 'MT OR ML']
    task_indices = [2, 6, 8, 10]
    
    # Collect all performance data
    task_data_matrix = []
    
    for env_type in range(4):
        try:
            data = dd.io.load(f'exps_data/exp3_bdqn_returns_{env_type}.h5')
            env_task_data = [data[:, idx] for idx in task_indices]
            if len(task_data_matrix) == 0:
                task_data_matrix = [[samples] for samples in env_task_data]
            else:
                for i, samples in enumerate(env_task_data):
                    task_data_matrix[i].append(samples)
        except FileNotFoundError:
            print(f"Warning: exp3_bdqn_returns_{env_type}.h5 not found")
            continue
    
    if not task_data_matrix:
        print("No data found for correlation analysis")
        return
    
    # Flatten each task's data across all environments
    flattened_data = []
    for task_idx in range(len(task_indices)):
        task_samples = []
        for env_samples in task_data_matrix[task_idx]:
            task_samples.extend(env_samples)
        flattened_data.append(task_samples)
    
    # Convert to DataFrame for correlation
    df = pd.DataFrame({name: data for name, data in zip(task_names, flattened_data)})
    
    # Calculate correlation matrix
    correlation_matrix = df.corr()
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    rc_ = {'axes.labelsize': 14, 'xtick.labelsize': 12, 
           'ytick.labelsize': 12, 'font.size': 12}
    sns.set(rc=rc_, style="whitegrid")
    
    # Create correlation heatmap
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Upper triangle mask
    
    ax = sns.heatmap(correlation_matrix, 
                     mask=mask,
                     annot=True, 
                     fmt='.3f',
                     cmap='RdBu_r',  # Blue-White-Red for correlations
                     center=0,
                     square=True,
                     vmin=-1, vmax=1,
                     cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title('MT/ML Task Performance Correlation Matrix', fontsize=16, pad=20)
    plt.xlabel('Tasks', fontsize=14)
    plt.ylabel('Tasks', fontsize=14)
    
    plt.tight_layout()
    plt.savefig("plots/mt_ml_correlation_matrix.pdf", bbox_inches='tight', dpi=300)
    plt.close()  # Close instead of show
    
    # Print correlation insights
    print("\n=== MT/ML Task Correlation Analysis ===")
    print("Correlation Matrix:")
    print(correlation_matrix.round(3))
    
    # Find highest and lowest correlations (excluding self-correlations)
    corr_values = []
    for i in range(len(task_names)):
        for j in range(i+1, len(task_names)):
            corr_values.append((correlation_matrix.iloc[i,j], task_names[i], task_names[j]))
    
    corr_values.sort(key=lambda x: x[0], reverse=True)
    
    print(f"\nHighest correlation: {corr_values[0][1]} ↔ {corr_values[0][2]} ({corr_values[0][0]:.3f})")
    print(f"Lowest correlation: {corr_values[-1][1]} ↔ {corr_values[-1][2]} ({corr_values[-1][0]:.3f})")


if __name__ == "__main__":
    print("Generating MT/ML heatmaps using BDQN policies from optimal Q-functions...")
    
    # Create plots directory if it doesn't exist
    import os
    os.makedirs("plots", exist_ok=True)
    
    # Generate visualizations
    print("1. Creating BDQN policy heatmaps from optimal Q-functions...")
    plot_majority_vote_policies()
    
    print("\n2. Creating policy arrow heatmaps for all 4 BDQN scenarios...")
    plot_policy_arrows()
    
    print("\n3. Creating performance heatmap...")
    plot_conjunction_disjunction_heatmap()
    
    print("\nAll heatmaps saved to plots/ directory:")
    print("BDQN policies from optimal Q-functions:")
    print("- plots/bdqn_policy_dense_rewards_same_absorbing_set.pdf")
    print("- plots/bdqn_policy_sparse_rewards_different_absorbing_set.pdf") 
    print("- plots/bdqn_policy_dense_rewards_different_absorbing_set.pdf")
    print("- plots/bdqn_policy_sparse_rewards_same_absorbing_set.pdf")
    print("Learned BDQN policies for evaluation:")
    print("- plots/bdqn_learned_policy_dense_rewards_same_absorbing_set.pdf")
    print("- plots/bdqn_learned_policy_sparse_rewards_different_absorbing_set.pdf") 
    print("- plots/bdqn_learned_policy_dense_rewards_different_absorbing_set.pdf")
    print("- plots/bdqn_learned_policy_sparse_rewards_same_absorbing_set.pdf")
    print("Performance heatmap:")
    print("- plots/mt_ml_heatmap.pdf")
    print("\nVisualization complete!")