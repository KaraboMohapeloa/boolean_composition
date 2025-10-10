import numpy as np
from collections import defaultdict
from copy import deepcopy

#########################################################################################
# Policy Improvement Functions for Bootstrapped DQN
#########################################################################################

def ensemble_policy_improvement(env, Q_list, method="average"):
    """
    Policy improvement for standard Q-learning ensemble
    
    Arguments:
    env -- environment with which agent interacts
    Q_list -- list of Q-functions (each Q is a dict mapping state to action-values)
    method -- 'average' (default) or 'majority' for action selection

    Returns:
    policy_improved -- Improved policy function
    """
    def policy_improved(state):
        n_actions = env.action_space.n
        q_values = []
        for Q in Q_list:
            if state in Q:
                q_values.append(Q[state])
        
        if not q_values:
            return np.ones(n_actions) / n_actions
            
        if method == "average":
            avg_q = np.mean(q_values, axis=0)
            best_action = np.argmax(avg_q)
        elif method == "majority":
            best_actions = [np.argmax(q) for q in q_values]
            counts = np.bincount(best_actions, minlength=n_actions)
            best_action = np.argmax(counts)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        probs = np.zeros(n_actions)
        probs[best_action] = 1.0
        return probs
    
    return policy_improved

def ensemble_gpi_policy_improvement(env, Q_list, method="average"):
    """
    Generalized policy improvement for goal-oriented Q-learning ensemble
    
    Arguments:
    env -- environment with which agent interacts
    Q_list -- list of Q-functions (each Q is a dict mapping state to dicts of goal: action-values)
    method -- 'average' (default) or 'majority' for action selection

    Returns:
    policy_improved -- Improved policy function (state, goal=None) -> action probabilities
    """
    def policy_improved(state, goal=None):
        n_actions = env.action_space.n
        q_values = []
        
        for Q in Q_list:
            if state in Q:
                if goal is not None and goal in Q[state]:
                    q_values.append(Q[state][goal])
                elif goal is None and Q[state]:
                    # GPI: max over goals for this state
                    head_q = np.max([Q[state][g] for g in Q[state].keys()], axis=0)
                    q_values.append(head_q)
        
        if not q_values:
            return np.ones(n_actions) / n_actions
            
        if method == "average":
            avg_q = np.mean(q_values, axis=0)
            best_action = np.argmax(avg_q)
        elif method == "majority":
            best_actions = [np.argmax(q) for q in q_values]
            counts = np.bincount(best_actions, minlength=n_actions)
            best_action = np.argmax(counts)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        probs = np.zeros(n_actions)
        probs[best_action] = 1.0
        return probs
    
    return policy_improved

#########################################################################################
# Equality Checking Functions
#########################################################################################

def Q_equal(Q1, Q2, epsilon=1e-3):    
    """Check if two Q-functions are equal within tolerance"""
    for state in Q1:
        if state not in Q2:
            return False
        for action in range(len(Q1[state])): 
            v1 = Q1[state][action]
            v2 = Q2[state][action]
            if abs(v1 - v2) > epsilon:
                return False
    return True

def EQ_equal(EQ1, EQ2, epsilon=1e-3):
    """Check if two EVFs are equal within tolerance"""
    for state in EQ1:
        if state not in EQ2:
            return False
        for goal in EQ1[state]:
            if goal not in EQ2[state]:
                return False
            for action in range(len(EQ1[state][goal])): 
                v1 = EQ1[state][goal][action]
                v2 = EQ2[state][goal][action]
                # Allow small tolerance or both very negative (unreachable) values
                if not (abs(v1 - v2) < epsilon or (v1 < -30 and v2 < -30)):
                    return False
    return True

#########################################################################################
# Bootstrapped DQN Algorithms
#########################################################################################

def Bootstrapped_Q_learning(env, Q_optimal=None, gamma=1, alpha=0.1, maxiter=300, 
                           n_heads=10, mask_prob=0.9, experience_replay_size=1000, 
                           batch_size=32, max_steps_per_episode=500, verbose=True):
    """
    Proper Bootstrapped DQN implementation for standard Q-learning
    
    Arguments:
    env -- environment with which agent interacts
    gamma -- discount factor
    alpha -- learning rate
    maxiter -- maximum number of episodes
    n_heads -- number of bootstrap heads
    mask_prob -- probability each head is active per episode
    experience_replay_size -- maximum replay buffer size per head
    batch_size -- number of transitions to sample for training
    max_steps_per_episode -- maximum steps before episode termination
    
    Returns:
    Q_list -- List of Q-heads
    stats -- Training statistics
    """
    # Initialize Q-heads and replay buffers
    Q_list = [defaultdict(lambda: np.zeros(env.action_space.n)) for _ in range(n_heads)]
    replay_buffers = [[] for _ in range(n_heads)]
    
    # Target networks for stable learning
    target_Q_list = [deepcopy(Q) for Q in Q_list]
    target_update_freq = 5  # Instead of 2
    
    # Use ensemble policy improvement
    behaviour_policy = ensemble_policy_improvement(env, Q_list, method="average")
    
    stats = {
        "R": [], 
        "T": 0,
        "steps_per_episode": [],
        "head_losses": [[] for _ in range(n_heads)],
        "ensemble_agreement_history": [],
        "ensemble_converged_at_episode": None,
        "episodes_terminated_by_step_limit": 0
    }
    
    k = 0
    T = 0
    state = env.reset()
    stats["R"].append(0)
    steps_in_episode = 0
    
    # Ensemble stopping criteria
    ensemble_threshold = 0.8  # Instead of 0.9 (too strict)
    sustained_required = 1  # Instead of 1 episodes
    sustained_count = 0
    
    while k < maxiter:
        # Use the ensemble policy to choose action
        probs = behaviour_policy(state)
        action = np.random.choice(np.arange(len(probs)), p=probs)
        
        # Take action
        state_, reward, done, _ = env.step(action)
        stats["R"][k] += reward
        T += 1
        steps_in_episode += 1
        
        # Store transition in replay buffers of active heads (based on masks)
        masks = np.random.binomial(1, mask_prob, size=n_heads)
        for i in range(n_heads):
            if masks[i]:
                replay_buffers[i].append((state, action, reward, state_, done))
                if len(replay_buffers[i]) > experience_replay_size:
                    replay_buffers[i].pop(0)
        
        # Train each head on its own replay buffer
        for i in range(n_heads):
            if masks[i] and len(replay_buffers[i]) >= batch_size:
                # Sample batch from this head's replay buffer
                batch_indices = np.random.choice(len(replay_buffers[i]), 
                                               size=min(batch_size, len(replay_buffers[i])), 
                                               replace=False)
                batch = [replay_buffers[i][idx] for idx in batch_indices]
                
                total_loss = 0
                for s, a, r, s_, d in batch:
                    # Bootstrap from this head's own target Q-values
                    if d:
                        target = r
                    else:
                        if s_ in target_Q_list[i]:
                            target = r + gamma * np.max(target_Q_list[i][s_])
                        else:
                            target = r
                    
                    # Q-learning update
                    td_error = target - Q_list[i][s][a]
                    Q_list[i][s][a] += alpha * td_error
                    total_loss += td_error ** 2
                
                stats["head_losses"][i].append(total_loss / len(batch))
        
        state = state_
        
        # Check if episode should terminate due to step limit
        step_limit_reached = steps_in_episode >= max_steps_per_episode
        episode_done = done or step_limit_reached
        
        if episode_done:
            stats["steps_per_episode"].append(steps_in_episode)
            
            if step_limit_reached and not done:
                stats["episodes_terminated_by_step_limit"] += 1
                if verbose:
                    print(f"Episode {k} terminated after {steps_in_episode} steps (max steps reached)")
            
            state = env.reset()
            stats["R"].append(0)
            k += 1
            steps_in_episode = 0
            
            # Update target networks periodically
            if k % target_update_freq == 0:
                target_Q_list = [deepcopy(Q) for Q in Q_list]
                # Update behaviour policy with new Q_list
                behaviour_policy = ensemble_policy_improvement(env, Q_list, method="average")
                if verbose:
                    print(f"Episode {k}: Updated target networks and behaviour policy")
            
            # Check ensemble agreement with optimal Q (if provided)
            if Q_optimal is not None:
                agreement = compute_ensemble_agreement(Q_list, Q_optimal, env.action_space.n)
                stats["ensemble_agreement_history"].append(agreement)
                
                if agreement >= ensemble_threshold:
                    sustained_count += 1
                else:
                    sustained_count = 0
                
                if sustained_count >= sustained_required:
                    stats["ensemble_converged_at_episode"] = k
                    if verbose:
                        print(f"Ensemble converged at episode {k} with agreement {agreement:.3f}")
                    break
            
            if verbose and k % 10 == 0:
                avg_reward = np.mean(stats["R"][-10:])
                avg_steps = np.mean(stats["steps_per_episode"][-10:]) if len(stats["steps_per_episode"]) >= 10 else steps_in_episode
                step_limit_percent = (stats["episodes_terminated_by_step_limit"] / k) * 100
                print(f"Episode {k}, Avg Reward: {avg_reward:.3f}, Avg Steps: {avg_steps:.1f}, Step Limits: {step_limit_percent:.1f}%")
    
    stats["T"] = T
    return Q_list, stats


def Bootstrapped_Goal_Oriented_Q_learning(env, T_states=None, Q_optimal=None, gamma=1, 
                                         alpha=0.1, maxiter=300, n_heads=10, mask_prob=0.9,
                                         experience_replay_size=1000, batch_size=32, 
                                         max_steps_per_episode=500, verbose=True):
    """
    Proper Bootstrapped DQN for goal-oriented Q-learning with Extended Value Functions
    
    Arguments:
    env -- environment with which agent interacts
    T_states -- terminal states to initialize goal memory
    gamma -- discount factor
    alpha -- learning rate
    maxiter -- maximum number of episodes
    n_heads -- number of bootstrap heads
    mask_prob -- probability each head is active per episode
    max_steps_per_episode -- maximum steps before episode termination
    
    Returns:
    Q_list -- List of goal-oriented Q-heads (EVFs)
    stats -- Training statistics
    """
    N = min(env.rmin, (env.rmin - env.rmax) * env.diameter)
    
    # Each head has its own Q-function and goal memory
    Q_list = [defaultdict(lambda: defaultdict(lambda: np.zeros(env.action_space.n))) 
              for _ in range(n_heads)]
    goal_memories = [set() for _ in range(n_heads)]
    
    # Initialize goal memories with terminal states
    if T_states:
        for state in T_states:
            state_str = str(state)
            for memory in goal_memories:
                memory.add(state_str)
    
    # Replay buffers for each head
    replay_buffers = [[] for _ in range(n_heads)]
    
    # Target networks
    target_Q_list = [deepcopy(Q) for Q in Q_list]
    target_update_freq = 5

    # Use ensemble GPI policy improvement
    behaviour_policy = ensemble_gpi_policy_improvement(env, Q_list, method="average")
    
    stats = {
        "R": [], 
        "T": 0,
        "steps_per_episode": [],
        "head_losses": [[] for _ in range(n_heads)],
        "ensemble_agreement_history": [],
        "ensemble_converged_at_episode": None,
        "goal_coverage": [],
        "episodes_terminated_by_step_limit": 0
    }
    
    k = 0
    T = 0
    state = env.reset()
    stats["R"].append(0)
    steps_in_episode = 0
    
    # Ensemble stopping criteria
    ensemble_threshold = 0.8
    sustained_required = 1  # Instead of 1 episodes
    sustained_count = 0
    
    while k < maxiter:
        # Use the ensemble GPI policy to choose action
        probs = behaviour_policy(state)  # No specific goal - uses GPI
        action = np.random.choice(np.arange(len(probs)), p=probs)
        
        # Take action
        state_, reward, done, _ = env.step(action)
        stats["R"][k] += reward
        T += 1
        steps_in_episode += 1
        
        # Store transition in replay buffers of active heads
        masks = np.random.binomial(1, mask_prob, size=n_heads)
        for i in range(n_heads):
            if masks[i]:
                replay_buffers[i].append((state, action, reward, state_, done))
                if len(replay_buffers[i]) > experience_replay_size:
                    replay_buffers[i].pop(0)
                
                # Add terminal state to this head's goal memory
                if done:
                    state_str = str(state_)
                    goal_memories[i].add(state_str)
        
        # Train each active head
        for i in range(n_heads):
            if masks[i] and len(replay_buffers[i]) >= batch_size:
                # Sample batch from this head's replay buffer
                batch_indices = np.random.choice(len(replay_buffers[i]), 
                                               size=min(batch_size, len(replay_buffers[i])), 
                                               replace=False)
                batch = [replay_buffers[i][idx] for idx in batch_indices]
                
                total_loss = 0
                update_count = 0
                
                for s, a, r, s_, d in batch:
                    # Update for all goals in this head's memory
                    for goal in goal_memories[i]:
                        # Extended reward calculation
                        if s != goal and d:
                            reward_ = N
                        else:
                            reward_ = r
                        
                        # Initialize Q-values if needed
                        if s not in Q_list[i]:
                            Q_list[i][s] = defaultdict(lambda: np.zeros(env.action_space.n))
                        if goal not in Q_list[i][s]:
                            Q_list[i][s][goal] = np.zeros(env.action_space.n)
                        
                        # Bootstrap from this head's own target values
                        if d:
                            target = reward_
                        else:
                            if s_ in target_Q_list[i] and goal in target_Q_list[i][s_]:
                                target = reward_ + gamma * np.max(target_Q_list[i][s_][goal])
                            else:
                                target = reward_
                        
                        # Q-learning update
                        td_error = target - Q_list[i][s][goal][a]
                        Q_list[i][s][goal][a] += alpha * td_error
                        total_loss += td_error ** 2
                        update_count += 1
                
                if update_count > 0:
                    stats["head_losses"][i].append(total_loss / update_count)
        
        state = state_
        
        # Check if episode should terminate due to step limit
        step_limit_reached = steps_in_episode >= max_steps_per_episode
        episode_done = done or step_limit_reached
        
        if episode_done:
            stats["steps_per_episode"].append(steps_in_episode)
            
            if step_limit_reached and not done:
                stats["episodes_terminated_by_step_limit"] += 1
                if verbose:
                    print(f"Episode {k} terminated after {steps_in_episode} steps (max steps reached)")
            
            state = env.reset()
            stats["R"].append(0)
            k += 1
            steps_in_episode = 0
            
            # Update target networks and behaviour policy
            if k % target_update_freq == 0:
                target_Q_list = [deepcopy(Q) for Q in Q_list]
                behaviour_policy = ensemble_gpi_policy_improvement(env, Q_list, method="average")
                if verbose:
                    print(f"Episode {k}: Updated target networks and behaviour policy")
            
            # Track goal coverage
            avg_goals = np.mean([len(memory) for memory in goal_memories])
            stats["goal_coverage"].append(avg_goals)
            
            # Check ensemble agreement with optimal EVF (if provided)
            if Q_optimal is not None:
                agreement = compute_evf_ensemble_agreement(Q_list, Q_optimal, env.action_space.n)
                stats["ensemble_agreement_history"].append(agreement)
                
                if agreement >= ensemble_threshold:
                    sustained_count += 1
                else:
                    sustained_count = 0
                
                if sustained_count >= sustained_required:
                    stats["ensemble_converged_at_episode"] = k
                    if verbose:
                        print(f"Ensemble converged at episode {k} with agreement {agreement:.3f}")
                    break
            
            if verbose and k % 10 == 0:
                avg_reward = np.mean(stats["R"][-10:])
                avg_goals = stats["goal_coverage"][-1] if stats["goal_coverage"] else 0
                avg_steps = np.mean(stats["steps_per_episode"][-10:]) if len(stats["steps_per_episode"]) >= 10 else steps_in_episode
                step_limit_percent = (stats["episodes_terminated_by_step_limit"] / k) * 100
                print(f"Episode {k}, Avg Reward: {avg_reward:.3f}, Avg Goals: {avg_goals:.1f}, Avg Steps: {avg_steps:.1f}, Step Limits: {step_limit_percent:.1f}%")
    
    stats["T"] = T
    return Q_list, stats

#########################################################################################
# Ensemble Agreement Computation
#########################################################################################

def compute_ensemble_agreement(Q_list, Q_optimal, n_actions, tolerance=1e-3):
    """Compute agreement between ensemble and optimal Q-function"""
    states = set()
    for Q in Q_list:
        states.update(Q.keys())
    
    if not states:
        return 0.0
    
    agree_count = 0
    total_count = 0
    
    for state in states:
        if state not in Q_optimal:
            continue
            
        # Get ensemble action (average Q-values across heads)
        ensemble_q = np.zeros(n_actions)
        head_count = 0
        for Q in Q_list:
            if state in Q:
                ensemble_q += Q[state]
                head_count += 1
        
        if head_count == 0:
            continue
            
        ensemble_q /= head_count
        ensemble_action = np.argmax(ensemble_q)
        
        # Check if ensemble action is optimal
        optimal_value = np.max(Q_optimal[state])
        optimal_actions = np.where(Q_optimal[state] >= optimal_value - tolerance)[0]
        
        if ensemble_action in optimal_actions:
            agree_count += 1
        total_count += 1
    
    return agree_count / total_count if total_count > 0 else 0.0


def compute_evf_ensemble_agreement(Q_list, EQ_optimal, n_actions, tolerance=1e-3):
    """Compute agreement for goal-oriented Q-functions"""
    states = set()
    for Q in Q_list:
        states.update(Q.keys())
    
    if not states:
        return 0.0
    
    agree_count = 0
    total_count = 0
    
    for state in states:
        if state not in EQ_optimal:
            continue
            
        # For EVF, we compare the GPI policy (max over goals)
        ensemble_evf = np.zeros(n_actions)
        head_count = 0
        
        for Q in Q_list:
            if state in Q and Q[state]:
                # GPI: max over goals
                head_evf = np.max([Q[state][goal] for goal in Q[state].keys()], axis=0)
                ensemble_evf += head_evf
                head_count += 1
        
        if head_count == 0:
            continue
            
        ensemble_evf /= head_count
        ensemble_action = np.argmax(ensemble_evf)
        
        # Compare with optimal GPI policy
        if EQ_optimal[state]:
            optimal_evf = np.max([EQ_optimal[state][goal] for goal in EQ_optimal[state].keys()], axis=0)
            optimal_value = np.max(optimal_evf)
            optimal_actions = np.where(optimal_evf >= optimal_value - tolerance)[0]
            
            if ensemble_action in optimal_actions:
                agree_count += 1
            total_count += 1
    
    return agree_count / total_count if total_count > 0 else 0.0

#########################################################################################
# Policy and Value Extraction Functions
#########################################################################################

def EQ_NP(EQ):
    """Extract goal-specific policies from EVF"""
    P = defaultdict(lambda: defaultdict(lambda: 0))
    for state in EQ:
        for goal in EQ[state]:
            P[state][goal] = np.argmax(EQ[state][goal])
    return P

def EQ_P(EQ, goal=None):
    """Extract policy from EVF (with or without specific goal)"""
    P = defaultdict(lambda: 0)
    for state in EQ:
        if goal:
            P[state] = np.argmax(EQ[state][goal])
        else:
            Vs = [EQ[state][goal] for goal in EQ[state].keys()]
            P[state] = np.argmax(np.max(Vs, axis=0))
    return P

def Q_P(Q):
    """Extract policy from Q-function"""
    P = defaultdict(lambda: 0)
    for state in Q:
        P[state] = np.argmax(Q[state])
    return P

def EQ_NV(EQ):
    """Extract goal-specific values from EVF"""
    V = defaultdict(lambda: defaultdict(lambda: 0))
    for state in EQ:
        for goal in EQ[state]:
            V[state][goal] = np.max(EQ[state][goal])
    return V

def EQ_V(EQ, goal=None):
    """Extract value function from EVF (with or without specific goal)"""
    V = defaultdict(lambda: 0)
    for state in EQ:
        if goal:
            V[state] = np.max(EQ[state][goal])
        else:
            Vs = [EQ[state][goal] for goal in EQ[state].keys()]
            V[state] = np.max(np.max(Vs, axis=0))
    return V

def NV_V(NV, goal=None):
    """Extract value function from nested value structure"""
    V = defaultdict(lambda: 0)
    for state in NV:
        if goal:
            V[state] = NV[state][goal]
        else:
            Vs = [NV[state][goal] for goal in NV[state].keys()]
            V[state] = np.max(Vs)
    return V

def Q_V(Q):
    """Extract value function from Q-function"""
    V = defaultdict(lambda: 0)
    for state in Q:
        V[state] = np.max(Q[state])
    return V

def EQ_Q(EQ, goal=None):
    """Convert EVF to Q-function (with or without specific goal)"""
    Q = defaultdict(lambda: np.zeros(5))  # Assuming 5 actions
    for state in EQ:
        if goal:
            Q[state] = EQ[state][goal]
        else:
            Vs = [EQ[state][goal] for goal in EQ[state].keys()]
            Q[state] = np.max(Vs, axis=0)
    return Q

#########################################################################################
# Composition Operations
#########################################################################################

def MAX(Q1, Q2):
    """Element-wise maximum of two Q-functions"""
    Q = defaultdict(lambda: np.zeros(5))
    for s in list(set(Q1.keys()) & set(Q2.keys())):
        Q[s] = np.max([Q1[s], Q2[s]], axis=0)
    return Q

def AVG(Q1, Q2):
    """Element-wise average of two Q-functions"""
    Q = defaultdict(lambda: np.zeros(5))
    for s in list(set(Q1.keys()) & set(Q2.keys())):
        Q[s] = (Q1[s] + Q2[s]) / 2
    return Q

def EQMAX(EQ, rmax=2):
    """Estimate maximum EVF bounds"""
    EQ_max = defaultdict(lambda: defaultdict(lambda: np.zeros(5)))
    for s in list(EQ.keys()):
        for g in list(EQ[s].keys()):
            c = rmax - max(EQ[g][g]) if g in EQ and g in EQ[g] else rmax
            if s == g:
                EQ_max[s][g] = EQ[s][g] * 0 + rmax
            else:      
                EQ_max[s][g] = EQ[s][g] + c   
    return EQ_max

def EQMIN(EQ, rmin=-0.1):
    """Estimate minimum EVF bounds"""
    EQ_min = defaultdict(lambda: defaultdict(lambda: np.zeros(5)))
    for s in list(EQ.keys()):
        for g in list(EQ[s].keys()):
            c = rmin - max(EQ[g][g]) if g in EQ and g in EQ[g] else rmin
            if s == g:
                EQ_min[s][g] = EQ[s][g] * 0 + rmin
            else:      
                EQ_min[s][g] = EQ[s][g] + c  
    return EQ_min

def NOT(EQ, EQ_max=None, EQ_min=None):
    """Negation operation for EVF"""
    EQ_max = EQ_max if EQ_max else EQMAX(EQ)
    EQ_min = EQ_min if EQ_min else EQMIN(EQ)
    EQ_not = defaultdict(lambda: defaultdict(lambda: np.zeros(5)))
    for s in list(EQ.keys()):
        for g in list(EQ[s].keys()):
            EQ_not[s][g] = (EQ_max[s][g] + EQ_min[s][g]) - EQ[s][g]    
    return EQ_not

def OR(EQ1, EQ2):
    """Disjunction operation for EVF"""
    EQ = defaultdict(lambda: defaultdict(lambda: np.zeros(5)))
    for s in list(EQ1.keys()):
        for g in list(EQ1[s].keys()):
            if s in EQ2 and g in EQ2[s]:
                EQ[s][g] = np.max([EQ1[s][g], EQ2[s][g]], axis=0)
    return EQ

def AND(EQ1, EQ2):
    """Conjunction operation for EVF"""
    EQ = defaultdict(lambda: defaultdict(lambda: np.zeros(5)))
    for s in list(EQ1.keys()):
        for g in list(EQ1[s].keys()):
            if s in EQ2 and g in EQ2[s]:
                EQ[s][g] = np.min([EQ1[s][g], EQ2[s][g]], axis=0)
    return EQ

#########################################################################################
# Ensemble Averaging Utilities
#########################################################################################

def average_ensemble_q(Q_list):
    """Average multiple Q-functions into one"""
    if not Q_list:
        return defaultdict(lambda: np.zeros(5))
    
    Q_avg = defaultdict(lambda: np.zeros(5))
    state_counts = defaultdict(int)
    
    for Q in Q_list:
        for state in Q:
            Q_avg[state] += Q[state]
            state_counts[state] += 1
    
    for state in Q_avg:
        if state_counts[state] > 0:
            Q_avg[state] /= state_counts[state]
    
    return Q_avg

def average_ensemble_evf(Q_list):
    """Average multiple EVFs into one"""
    if not Q_list:
        return defaultdict(lambda: defaultdict(lambda: np.zeros(5)))
    
    EQ_avg = defaultdict(lambda: defaultdict(lambda: np.zeros(5)))
    state_goal_counts = defaultdict(lambda: defaultdict(int))
    
    for Q in Q_list:
        for state in Q:
            for goal in Q[state]:
                EQ_avg[state][goal] += Q[state][goal]
                state_goal_counts[state][goal] += 1
    
    for state in EQ_avg:
        for goal in EQ_avg[state]:
            if state_goal_counts[state][goal] > 0:
                EQ_avg[state][goal] /= state_goal_counts[state][goal]
    
    return EQ_avg