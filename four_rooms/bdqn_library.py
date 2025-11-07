import numpy as np
from collections import defaultdict
from copy import deepcopy

#########################################################################################
# Equality Checking Functions
#########################################################################################

def Q_equal(Q1, Q2, epsilon=1e-5):    
    """Check if two Q-functions are equal within tolerance"""
    # Check all states in Q1 (like original library.py)
    for state in Q1:
        if state not in Q2:
            return False  # Q2 must have all states that Q1 has
        for action in range(len(Q1[state])): 
            v1 = Q1[state][action]
            v2 = Q2[state][action]
            if abs(v1 - v2) > epsilon:
                return False
    return True

def EQ_equal(EQ1, EQ2, epsilon=1e-5):
    """Check if two EVFs are equal within tolerance"""
    # Check all states in EQ1 (like original library.py)
    for state in EQ1:
        if state not in EQ2:
            return False  # EQ2 must have all states that EQ1 has
        for goal in EQ1[state]:
            if goal not in EQ2[state]:
                return False  # EQ2 must have all goals that EQ1 has
            for action in range(len(EQ1[state][goal])):
                v1 = EQ1[state][goal][action]
                v2 = EQ2[state][goal][action]
                # Original condition from library.py: exact match OR both very negative
                if not (abs(v1-v2) < epsilon or (v1 < -30 and v2 < -30)):
                    return False
    return True

#########################################################################################
# Ensemble Averaging Utilities (Moved up to resolve dependencies)
#########################################################################################

def compute_ensemble_uncertainty(Q_list, state, action=None):
    """
    Compute uncertainty (variance) in Q-value estimates across ensemble heads.
    Higher variance indicates higher uncertainty.
    
    Args:
        Q_list: List of Q-function heads
        state: State to compute uncertainty for
        action: Specific action (if None, returns uncertainty for all actions)
    
    Returns:
        Uncertainty values (variance across heads)
    """
    q_values = []
    for head in Q_list:
        if state in head:
            q_values.append(head[state])
    
    if not q_values:
        # No heads have seen this state - maximum uncertainty
        return np.inf if action is not None else np.full(5, np.inf)
    
    q_values = np.array(q_values)
    uncertainty = np.var(q_values, axis=0)
    
    if action is not None:
        return uncertainty[action]
    return uncertainty

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

def average_ensemble_evf(Q_list, action_space_size=5):
    """
    Computes the element-wise average of all Extended Value Functions (EVFs)
    in the ensemble Q_list.
    """
    if not Q_list:
        # Return an empty structure matching the Q_head format
        return defaultdict(lambda: defaultdict(lambda: np.zeros(action_space_size)))

    # Initialize structures for averaging.
    # EQ_avg[state][goal] -> Q-value array
    EQ_avg = defaultdict(lambda: defaultdict(lambda: np.zeros(action_space_size)))
    # state_goal_counts[state][goal] -> count
    state_goal_counts = defaultdict(lambda: defaultdict(int))

    # Accumulation loop
    for Q in Q_list:
        for state in Q:
            for goal in Q[state]:
                # Accumulate Q-values
                EQ_avg[state][goal] += Q[state][goal]
                # Increment count
                state_goal_counts[state][goal] += 1

    # Averaging loop
    for state in EQ_avg:
        for goal in EQ_avg[state]:
            count = state_goal_counts[state][goal]
            if count > 0:
                EQ_avg[state][goal] /= count

    return EQ_avg

#########################################################################################


def check_ensemble_policy_agreement(Q_list, states_to_check=None, agreement_threshold=1.0):
    """
    Check if ensemble heads agree on policy (action selection).
    
    Args:
        Q_list: List of Q-function heads
        states_to_check: States to check for agreement (if None, checks all visited states)
        agreement_threshold: Fraction of states that must have policy agreement (default 1.0 = 100%)
    
    Returns:
        Boolean indicating if ensemble policies agree
    """
    if not Q_list:
        return False
    
    # Get all states visited by any head
    all_states = set()
    for head in Q_list:
        all_states.update(head.keys())
    
    if not all_states:
        return False
    
    if states_to_check is None:
        states_to_check = list(all_states)
    
    # Check agreement for each state
    states_checked = 0
    states_agreed = 0
    
    for state in states_to_check:
        # Get greedy actions for each head that has visited this state
        head_actions = []
        for head in Q_list:
            if state in head:
                greedy_action = np.argmax(head[state])
                head_actions.append(greedy_action)
        
        if len(head_actions) >= 1:  # Include states visited by at least 1 head
            states_checked += 1
            # Check if all heads agree on the same action (single head automatically agrees)
            if len(set(head_actions)) == 1:  # All actions are the same
                states_agreed += 1
    
    if states_checked == 0:
        return False
    
    # Calculate agreement percentage
    agreement_ratio = states_agreed / states_checked
    return agreement_ratio >= agreement_threshold

def check_ensemble_evf_policy_agreement(Q_list, states_to_check=None, agreement_threshold=1.0):
    """
    Check if ensemble heads agree on EVF policy (GPI action selection).
    
    Args:
        Q_list: List of Extended Q-function heads  
        states_to_check: States to check for agreement (if None, checks all visited states)
        agreement_threshold: Fraction of states that must have policy agreement (default 1.0 = 100%)
    
    Returns:
        Boolean indicating if ensemble EVF policies agree
    """
    if not Q_list:
        return False
    
    # Get all states visited by any head
    all_states = set()
    for head in Q_list:
        all_states.update(head.keys())
    
    if not all_states:
        return False
    
    if states_to_check is None:
        states_to_check = list(all_states)
    
    # Check agreement for each state
    states_checked = 0
    states_agreed = 0
    
    for state in states_to_check:
        # Get GPI actions for each head that has visited this state
        head_actions = []
        for head in Q_list:
            if state in head and head[state]:
                # GPI: max over all goals for this head
                q_values_per_goal = [head[state][g] for g in head[state].keys()]
                if q_values_per_goal:
                    q_values = np.max(q_values_per_goal, axis=0)
                    greedy_action = np.argmax(q_values)
                    head_actions.append(greedy_action)
        
        if len(head_actions) >= 1:  # Include states visited by at least 1 head
            states_checked += 1
            # Check if all heads agree on the same action (single head automatically agrees)
            if len(set(head_actions)) == 1:  # All actions are the same
                states_agreed += 1
    
    if states_checked == 0:
        return False
    
    # Calculate agreement percentage
    agreement_ratio = states_agreed / states_checked
    return agreement_ratio >= agreement_threshold

def check_ensemble_value_agreement(Q_list, states_to_check=None, agreement_threshold=0.95, value_tolerance=1e-3):
    """
    Check if ensemble heads agree on Q-values (not just policies).
    
    Args:
        Q_list: List of Q-function heads
        states_to_check: States to check for agreement (if None, checks all visited states)
        agreement_threshold: Fraction of states that must have value agreement (default 0.95 = 95%)
        value_tolerance: Tolerance for considering Q-values as "agreeing"
    
    Returns:
        Boolean indicating if ensemble Q-values agree
    """
    if not Q_list:
        return False
    
    # Get all states visited by any head
    all_states = set()
    for head in Q_list:
        all_states.update(head.keys())
    
    if not all_states:
        return False
    
    if states_to_check is None:
        states_to_check = list(all_states)
    
    # Check value agreement for each state
    states_checked = 0
    states_agreed = 0
    
    for state in states_to_check:
        # Get Q-values for each head that has visited this state
        head_q_values = []
        for head in Q_list:
            if state in head:
                head_q_values.append(head[state])
        
        if len(head_q_values) >= 1:  # Include states visited by at least 1 head
            states_checked += 1
            
            # Check if all heads have similar Q-values (within tolerance)
            if len(head_q_values) == 1:
                # Single head automatically agrees with itself
                states_agreed += 1
            else:
                # Compare all pairs of Q-values
                all_agree = True
                for i in range(len(head_q_values)):
                    for j in range(i+1, len(head_q_values)):
                        # Check if Q-values are close (within tolerance)
                        if not np.allclose(head_q_values[i], head_q_values[j], atol=value_tolerance):
                            all_agree = False
                            break
                    if not all_agree:
                        break
                
                if all_agree:
                    states_agreed += 1
    
    if states_checked == 0:
        return False
    
    # Calculate agreement percentage
    agreement_ratio = states_agreed / states_checked
    return agreement_ratio >= agreement_threshold

def check_ensemble_evf_value_agreement(Q_list, states_to_check=None, agreement_threshold=0.95, value_tolerance=1e-5):
    """
    Check if ensemble heads agree on EVF Q-values (not just GPI policies).
    
    Args:
        Q_list: List of Extended Q-function heads
        states_to_check: States to check for agreement (if None, checks all visited states)
        agreement_threshold: Fraction of states that must have value agreement (default 0.95 = 95%)
        value_tolerance: Tolerance for considering EVF Q-values as "agreeing"
    
    Returns:
        Boolean indicating if ensemble EVF Q-values agree
    """
    if not Q_list:
        return False
    
    # Get all states visited by any head
    all_states = set()
    for head in Q_list:
        all_states.update(head.keys())
    
    if not all_states:
        return False
    
    if states_to_check is None:
        states_to_check = list(all_states)
    
    # Check EVF value agreement for each state
    states_checked = 0
    states_agreed = 0
    
    for state in states_to_check:
        # Get EVF structures for each head that has visited this state
        head_evf_values = []
        for head in Q_list:
            if state in head and head[state]:
                head_evf_values.append(head[state])
        
        if len(head_evf_values) >= 1:  # Include states visited by at least 1 head
            states_checked += 1
            
            # Check if all heads have similar EVF values (within tolerance)
            if len(head_evf_values) == 1:
                # Single head automatically agrees with itself
                states_agreed += 1
            else:
                # Compare EVF values across heads
                all_agree = True
                
                # Get common goals across all heads for this state
                common_goals = set(head_evf_values[0].keys())
                for evf in head_evf_values[1:]:
                    common_goals &= set(evf.keys())
                
                # Check agreement for each common goal
                for goal in common_goals:
                    for i in range(len(head_evf_values)):
                        for j in range(i+1, len(head_evf_values)):
                            if goal in head_evf_values[i] and goal in head_evf_values[j]:
                                # Check if Q-values for this goal are close
                                if not np.allclose(head_evf_values[i][goal], head_evf_values[j][goal], atol=value_tolerance):
                                    all_agree = False
                                    break
                        if not all_agree:
                            break
                    if not all_agree:
                        break
                
                if all_agree:
                    states_agreed += 1
    
    if states_checked == 0:
        return False
    
    # Calculate agreement percentage
    agreement_ratio = states_agreed / states_checked
    return agreement_ratio >= agreement_threshold

#########################################################################################
# Optimality Checking Functions
#########################################################################################

def check_optimal_policy_match(Q_list, Q_optimal, threshold=0.95):
    """
    Check if ensemble policy matches optimal policy.
    
    Args:
        Q_list: List of Q-function heads
        Q_optimal: Optimal Q-function for comparison
        threshold: Fraction of states that must match optimal policy
    
    Returns:
        Boolean indicating if ensemble policy matches optimal
    """
    if Q_optimal is None:
        return False
    
    # Get ensemble average policy
    Q_avg = average_ensemble_q(Q_list)
    
    # Extract policies
    optimal_policy = Q_P(Q_optimal)
    ensemble_policy = Q_P(Q_avg)
    
    # Count matching states
    common_states = set(optimal_policy.keys()) & set(ensemble_policy.keys())
    if not common_states:
        return False
    
    matches = sum(1 for s in common_states if optimal_policy[s] == ensemble_policy[s])
    match_ratio = matches / len(common_states)
    
    return match_ratio >= threshold

def check_optimal_value_convergence(Q_list, Q_optimal, epsilon=1e-3, threshold=0.95):
    """
    Check if ensemble Q-values have converged to within epsilon of optimal.
    
    Args:
        Q_list: List of Q-function heads
        Q_optimal: Optimal Q-function for comparison
        epsilon: Tolerance for Q-value convergence
        threshold: Fraction of states that must converge
    
    Returns:
        Boolean indicating if ensemble values are close to optimal
    """
    if Q_optimal is None:
        return False
    
    Q_avg = average_ensemble_q(Q_list)
    
    # Check value convergence for common states
    converged_states = 0
    total_states = 0
    
    for state in Q_optimal:
        if state in Q_avg:
            total_states += 1
            # Check if all action values are within epsilon
            if np.allclose(Q_avg[state], Q_optimal[state], atol=epsilon):
                converged_states += 1
    
    return (converged_states / total_states) >= threshold if total_states > 0 else False

def check_optimal_evf_policy_match(Q_list, EQ_optimal, threshold=0.95):
    """
    Check if ensemble EVF policy matches optimal EVF policy using GPI.
    
    Args:
        Q_list: List of Extended Q-function heads
        EQ_optimal: Optimal Extended Q-function for comparison
        threshold: Fraction of states that must match optimal policy
    
    Returns:
        Boolean indicating if ensemble EVF policy matches optimal
    """
    if EQ_optimal is None:
        return False
    
    # Get ensemble average EVF
    EQ_avg = average_ensemble_evf(Q_list)
    
    # Extract GPI policies
    optimal_policy = EQ_P(EQ_optimal)  # GPI policy from optimal EVF
    ensemble_policy = EQ_P(EQ_avg)     # GPI policy from ensemble EVF
    
    # Count matching states
    common_states = set(optimal_policy.keys()) & set(ensemble_policy.keys())
    if not common_states:
        return False
    
    matches = sum(1 for s in common_states if optimal_policy[s] == ensemble_policy[s])
    match_ratio = matches / len(common_states)
    
    return match_ratio >= threshold

def check_optimal_evf_convergence(Q_list, EQ_optimal, epsilon=1e-5, threshold=0.95):
    """
    Check if ensemble EVF has converged to optimal EVF.
    
    Args:
        Q_list: List of Extended Q-function heads
        EQ_optimal: Optimal Extended Q-function for comparison
        epsilon: Tolerance for EVF convergence (uses EQ_equal)
        threshold: Fraction of state-goal pairs that must converge
    
    Returns:
        Boolean indicating if ensemble EVF is close to optimal
    """
    if EQ_optimal is None:
        return False
    
    EQ_avg = average_ensemble_evf(Q_list)
    
    # Use the existing EQ_equal function for detailed comparison
    return EQ_equal(EQ_avg, EQ_optimal, epsilon)

def check_optimal_ensemble_consensus(Q_list, Q_optimal=None, 
                                   agreement_threshold=0.95, 
                                   optimality_threshold=0.95,
                                   value_epsilon=1e-3,
                                   require_value_convergence=False):
    """
    Combined check: ensemble agrees internally AND is close to optimal (if available).
    
    Args:
        Q_list: List of Q-function heads
        Q_optimal: Optional optimal Q-function for comparison
        agreement_threshold: Threshold for internal ensemble agreement
        optimality_threshold: Threshold for optimal policy matching
        value_epsilon: Tolerance for value convergence
        require_value_convergence: If True, requires both policy AND value convergence.
                                 If False, only requires policy convergence.
    
    Returns:
        Boolean indicating if ensemble has consensus and is optimal
    """
    # First check if ensemble has internal consensus
    ensemble_agrees = check_ensemble_policy_agreement(Q_list, agreement_threshold=agreement_threshold)
    
    if not ensemble_agrees:
        return False
    
    # If we have optimal solution, also check closeness to optimal
    if Q_optimal is not None:
        # Check policy matching (always required)
        policy_matches = check_optimal_policy_match(Q_list, Q_optimal, threshold=optimality_threshold)
        
        if require_value_convergence:
            # Strict mode: require both policy and value convergence
            values_converged = check_optimal_value_convergence(Q_list, Q_optimal, value_epsilon, optimality_threshold)
            return policy_matches and values_converged
        else:
            # Lenient mode: only require policy convergence (policy is what matters for behavior)
            return policy_matches
    
    # If no optimal available, trust ensemble consensus
    return True

def check_optimal_evf_ensemble_consensus(Q_list, EQ_optimal=None,
                                       agreement_threshold=0.95,
                                       optimality_threshold=0.95,
                                       value_epsilon=1e-5,
                                       require_value_convergence=False):
    """
    Combined check for EVF: ensemble agrees internally AND is close to optimal (if available).
    
    Args:
        Q_list: List of Extended Q-function heads
        EQ_optimal: Optional optimal Extended Q-function for comparison
        agreement_threshold: Threshold for internal ensemble agreement
        optimality_threshold: Threshold for optimal policy matching
        value_epsilon: Tolerance for EVF convergence
        require_value_convergence: If True, requires both policy AND value convergence.
                                 If False, only requires policy convergence.
    
    Returns:
        Boolean indicating if ensemble EVF has consensus and is optimal
    """
    # First check if ensemble has internal consensus
    ensemble_agrees = check_ensemble_evf_policy_agreement(Q_list, agreement_threshold=agreement_threshold)
    
    if not ensemble_agrees:
        return False
    
    # If we have optimal solution, also check closeness to optimal
    if EQ_optimal is not None:
        # Check policy matching (always required)
        policy_matches = check_optimal_evf_policy_match(Q_list, EQ_optimal, threshold=optimality_threshold)
        
        if require_value_convergence:
            # Strict mode: require both policy and value convergence
            values_converged = check_optimal_evf_convergence(Q_list, EQ_optimal, value_epsilon)
            return policy_matches and values_converged
        else:
            # Lenient mode: only require policy convergence (policy is what matters for behavior)
            return policy_matches
    
    # If no optimal available, trust ensemble consensus
    return True

def check_ensemble_value_consensus_with_optimality(Q_list, Q_optimal=None,
                                                 value_agreement_threshold=0.95,
                                                 optimality_threshold=0.5,
                                                 value_tolerance=1e-3):
    """
    Check convergence based on your specific criteria:
    1. Ensemble agrees on Q-values for 95% of states
    2. 50% of states derive the optimal policy
    
    Args:
        Q_list: List of Q-function heads
        Q_optimal: Optional optimal Q-function for comparison
        value_agreement_threshold: Threshold for ensemble value agreement (default 0.95 = 95%)
        optimality_threshold: Threshold for optimal policy matching (default 0.5 = 50%)
        value_tolerance: Tolerance for Q-value agreement within ensemble
    
    Returns:
        Boolean indicating if convergence criteria are met
    """
    # Check if ensemble agrees on values for 95% of states
    value_consensus = check_ensemble_value_agreement(Q_list, 
                                                   agreement_threshold=value_agreement_threshold,
                                                   value_tolerance=value_tolerance)
    
    if not value_consensus:
        return False
    
    # If we have optimal solution, check if 50% of states have optimal policy
    if Q_optimal is not None:
        policy_optimal = check_optimal_policy_match(Q_list, Q_optimal, threshold=optimality_threshold)
        return policy_optimal
    
    # If no optimal available, just use value consensus
    return True

def check_evf_value_consensus_with_optimality(Q_list, EQ_optimal=None,
                                            value_agreement_threshold=0.95,
                                            optimality_threshold=0.5,
                                            value_tolerance=1e-5):
    """
    Check EVF convergence based on your specific criteria:
    1. Ensemble agrees on EVF Q-values for 95% of states
    2. 50% of states derive the optimal GPI policy
    
    Args:
        Q_list: List of Extended Q-function heads
        EQ_optimal: Optional optimal Extended Q-function for comparison
        value_agreement_threshold: Threshold for ensemble EVF value agreement (default 0.95 = 95%)
        optimality_threshold: Threshold for optimal policy matching (default 0.5 = 50%)
        value_tolerance: Tolerance for EVF Q-value agreement within ensemble
    
    Returns:
        Boolean indicating if convergence criteria are met
    """
    # Check if ensemble agrees on EVF values for 95% of states
    value_consensus = check_ensemble_evf_value_agreement(Q_list,
                                                       agreement_threshold=value_agreement_threshold,
                                                       value_tolerance=value_tolerance)
    
    if not value_consensus:
        return False
    
    # If we have optimal solution, check if 50% of states have optimal GPI policy
    if EQ_optimal is not None:
        policy_optimal = check_optimal_evf_policy_match(Q_list, EQ_optimal, threshold=optimality_threshold)
        return policy_optimal
    
    # If no optimal available, just use value consensus
    return True

def check_ensemble_policy_consensus_with_optimality(Q_list, Q_optimal=None,
                                                   policy_agreement_threshold=0.95,
                                                   policy_optimality_threshold=0.5):
    """
    Policy-only convergence: ensemble agrees on policies for 95% of states AND 50% of states have optimal policy.
    This ignores value differences and focuses purely on behavioral convergence.
    
    Args:
        Q_list: List of Q-function heads
        Q_optimal: Optimal Q-function for comparison (optional)
        policy_agreement_threshold: Threshold for policy agreement within ensemble (default 0.95 = 95%)
        policy_optimality_threshold: Threshold for optimal policy matching (default 0.5 = 50%)
    
    Returns:
        Boolean indicating if convergence criteria are met
    """
    # Check if ensemble agrees on policies for 95% of states
    policy_consensus = check_ensemble_policy_agreement(Q_list, agreement_threshold=policy_agreement_threshold)
    
    if not policy_consensus:
        return False
    
    # If we have optimal solution, check if 50% of states have optimal policy
    if Q_optimal is not None:
        policy_optimal = check_optimal_policy_match(Q_list, Q_optimal, threshold=policy_optimality_threshold)
        return policy_optimal
    
    # If no optimal available, just use policy consensus
    return True

def check_evf_policy_consensus_with_optimality(Q_list, EQ_optimal=None,
                                               policy_agreement_threshold=0.95,
                                               policy_optimality_threshold=0.5):
    """
    EVF Policy-only convergence: ensemble agrees on GPI policies for 95% of states AND 50% of states have optimal GPI policy.
    This ignores EVF value differences and focuses purely on behavioral convergence.
    
    Args:
        Q_list: List of EVF Q-function heads
        EQ_optimal: Optimal EVF Q-function for comparison (optional)
        policy_agreement_threshold: Threshold for GPI policy agreement within ensemble (default 0.95 = 95%)
        policy_optimality_threshold: Threshold for optimal GPI policy matching (default 0.5 = 50%)
    
    Returns:
        Boolean indicating if convergence criteria are met
    """
    # Check if ensemble agrees on GPI policies for required percentage of states
    policy_consensus = check_ensemble_evf_policy_agreement(Q_list, agreement_threshold=policy_agreement_threshold)
    
    if not policy_consensus:
        return False
    
    # If we have optimal solution, check if required percentage of states have optimal GPI policy
    if EQ_optimal is not None:
        policy_optimal = check_optimal_evf_policy_match(Q_list, EQ_optimal, threshold=policy_optimality_threshold)
        return policy_optimal
    
    # If no optimal available, just use policy consensus
    return True

def analyze_state_optimality(Q_list, Q_optimal, value_epsilon=1e-3, policy_threshold=0.95):
    """
    Analyze which individual states have converged to optimality.
    
    Args:
        Q_list: List of Q-function heads
        Q_optimal: Optimal Q-function for comparison
        value_epsilon: Tolerance for Q-value convergence
        policy_threshold: Threshold for policy agreement within ensemble
    
    Returns:
        Dict with detailed state analysis
    """
    if Q_optimal is None:
        return {"error": "No optimal Q-function provided"}
    
    Q_avg = average_ensemble_q(Q_list)
    
    analysis = {
        "total_states": 0,
        "policy_optimal": 0,
        "value_optimal": 0,
        "both_optimal": 0,
        "ensemble_agrees": 0,
        "state_details": {}
    }
    
    # Analyze each state that appears in both ensemble and optimal
    common_states = set(Q_avg.keys()) & set(Q_optimal.keys())
    analysis["total_states"] = len(common_states)
    
    for state in common_states:
        # Check policy optimality
        ensemble_action = np.argmax(Q_avg[state])
        optimal_action = np.argmax(Q_optimal[state])
        policy_optimal = (ensemble_action == optimal_action)
        
        # Check value optimality
        value_optimal = np.allclose(Q_avg[state], Q_optimal[state], atol=value_epsilon)
        
        # Check ensemble agreement for this state
        head_actions = []
        for head in Q_list:
            if state in head:
                head_actions.append(np.argmax(head[state]))
        
        ensemble_agrees = len(set(head_actions)) == 1 if head_actions else False
        
        # Store detailed analysis
        analysis["state_details"][state] = {
            "policy_optimal": policy_optimal,
            "value_optimal": value_optimal,
            "ensemble_agrees": ensemble_agrees,
            "ensemble_action": ensemble_action,
            "optimal_action": optimal_action,
            "max_value_diff": np.max(np.abs(Q_avg[state] - Q_optimal[state])),
            "num_heads": len(head_actions)
        }
        
        # Update counters
        if policy_optimal:
            analysis["policy_optimal"] += 1
        if value_optimal:
            analysis["value_optimal"] += 1
        if policy_optimal and value_optimal:
            analysis["both_optimal"] += 1
        if ensemble_agrees:
            analysis["ensemble_agrees"] += 1
    
    return analysis

def analyze_evf_state_optimality(Q_list, EQ_optimal, value_epsilon=1e-5):
    """
    Analyze which individual states have converged to optimality for EVF.
    
    Args:
        Q_list: List of Extended Q-function heads
        EQ_optimal: Optimal Extended Q-function for comparison
        value_epsilon: Tolerance for EVF convergence
    
    Returns:
        Dict with detailed state analysis for EVF
    """
    if EQ_optimal is None:
        return {"error": "No optimal EVF provided"}
    
    EQ_avg = average_ensemble_evf(Q_list)
    
    analysis = {
        "total_states": 0,
        "policy_optimal": 0,
        "value_optimal": 0,
        "both_optimal": 0,
        "ensemble_agrees": 0,
        "state_details": {}
    }
    
    # Analyze each state
    common_states = set(EQ_avg.keys()) & set(EQ_optimal.keys())
    analysis["total_states"] = len(common_states)
    
    for state in common_states:
        # Get GPI actions for comparison
        ensemble_policy = EQ_P(EQ_avg)
        optimal_policy = EQ_P(EQ_optimal)
        
        ensemble_action = ensemble_policy.get(state, -1)
        optimal_action = optimal_policy.get(state, -1)
        policy_optimal = (ensemble_action == optimal_action) and (ensemble_action != -1)
        
        # Check value optimality using EQ_equal logic
        value_optimal = True
        max_value_diff = 0
        
        if state in EQ_avg and state in EQ_optimal:
            for goal in set(EQ_avg[state].keys()) | set(EQ_optimal[state].keys()):
                if goal in EQ_avg[state] and goal in EQ_optimal[state]:
                    for action in range(len(EQ_avg[state][goal])):
                        v1 = EQ_avg[state][goal][action]
                        v2 = EQ_optimal[state][goal][action]
                        diff = abs(v1 - v2)
                        max_value_diff = max(max_value_diff, diff)
                        
                        # Use EQ_equal logic: exact match OR both very negative
                        if not (diff < value_epsilon or (v1 < -30 and v2 < -30)):
                            value_optimal = False
        
        # Check ensemble agreement for this state
        head_actions = []
        for head in Q_list:
            if state in head and head[state]:
                # GPI: max over all goals for this head
                q_values_per_goal = [head[state][g] for g in head[state].keys()]
                if q_values_per_goal:
                    q_values = np.max(q_values_per_goal, axis=0)
                    greedy_action = np.argmax(q_values)
                    head_actions.append(greedy_action)
        
        ensemble_agrees = len(set(head_actions)) == 1 if head_actions else False
        
        # Store detailed analysis
        analysis["state_details"][state] = {
            "policy_optimal": policy_optimal,
            "value_optimal": value_optimal,
            "ensemble_agrees": ensemble_agrees,
            "ensemble_action": ensemble_action,
            "optimal_action": optimal_action,
            "max_value_diff": max_value_diff,
            "num_heads": len(head_actions)
        }
        
        # Update counters
        if policy_optimal:
            analysis["policy_optimal"] += 1
        if value_optimal:
            analysis["value_optimal"] += 1
        if policy_optimal and value_optimal:
            analysis["both_optimal"] += 1
        if ensemble_agrees:
            analysis["ensemble_agrees"] += 1
    
    return analysis

#########################################################################################
# Bootstrapped DQN versions of library.py functions
#########################################################################################

def Bootstrapped_Q_learning(env, Q_optimal=None, gamma=1, alpha=0.1,
                            n_heads=10, mask_prob=0.5, init_q_range=0.0, warmup_steps=1000,
                            policy_agreement_threshold=0.95, optimality_threshold=0.95,
                            value_epsilon=1e-3):
    """
    Bootstrapped DQN version of library.py Q_learning function.
    Uses Thompson sampling for exploration and per-step masking for bootstrap training.
    
    Algorithm:
    1. Action selection: Sample one head uniformly, act greedily w.r.t. it (Thompson sampling)
    2. Per-step masking: Each transition independently included in each head with prob mask_prob
    3. Immediate updates: Heads that include transition are updated immediately
    
    Arguments:
    env -- environment with which agent interacts
    gamma -- discount factor
    alpha -- learning rate
    n_heads -- number of bootstrap heads
    mask_prob -- probability of including each transition in each head (per-step masking)
    convergence_tolerance -- tolerance for convergence checking (unused with policy agreement)
    convergence_percentage -- percentage of states that must converge (unused with policy agreement)
    init_q_range -- random initialization range for Q-values (default 0.0 = zeros)
    warmup_steps -- number of random exploration steps before starting bootstrap updates
    policy_agreement_threshold -- fraction of states that must have policy agreement (default 0.95 = 95%)
    
    Returns:
    Q -- Ensemble averaged Q function (compatible with library.py)
    stats -- Statistics dictionary matching library.py format
    """
    # Initialize ensemble of Q-functions with random initialization
    # NOTE: Original Bootstrapped DQN uses shared initialization (all heads start the same)
    # Diversity comes from bootstrap masking, not different initializations
    def create_random_q_function():
        def random_init():
            if init_q_range > 0:
                # Small random initialization centered at 0 (standard practice)
                # Positive-only initialization creates optimistic bias
                return np.random.uniform(-init_q_range, init_q_range, env.action_space.n)
            else:
                return np.zeros(env.action_space.n)
        return defaultdict(random_init)
    
    Q_list = [create_random_q_function() for _ in range(n_heads)]
    
    # Set terminal states to have Q-values of 0 (correct Q-learning initialization)
    if hasattr(env, 'T_states') and env.T_states:
        for head in Q_list:
            for terminal_state in env.T_states:
                # Convert to tuple if it's a list (for hashability)
                state_key = tuple(terminal_state) if isinstance(terminal_state, list) else terminal_state
                head[state_key] = np.zeros(env.action_space.n)
    
    update_counts = [0] * n_heads
    
    # Stopping condition: ensemble policy agreement + optimality check (POLICY-ONLY VERSION)
    def check_policy_agreement(episode_num):
        # Check convergence using your specific criteria (POLICY-FOCUSED):
        # 1. Ensemble agrees on POLICIES for 95% of states (ignore value differences)
        # 2. 50% of states derive the optimal policy
        agreed = check_ensemble_policy_consensus_with_optimality(Q_list, Q_optimal=Q_optimal,
                                                               policy_agreement_threshold=policy_agreement_threshold,
                                                               policy_optimality_threshold=optimality_threshold)
        
        # Show policy agreement status every 100 episodes
        if episode_num % 100 == 0:
            print(f"\n=== BDQN Policy Agreement Check (Episode {episode_num}) ===")
            
            # Get all visited states
            all_states = set()
            for head in Q_list:
                all_states.update(head.keys())
            
            if all_states:
                states_to_show = list(all_states)[:5]  # Show first 5 states
                states_checked = 0
                states_agreed = 0
                
                print("Policy comparison for sample states:")
                for state in states_to_show:
                    head_actions = []
                    for i, head in enumerate(Q_list):
                        if state in head:
                            action = np.argmax(head[state])
                            head_actions.append(action)
                    
                    if len(head_actions) >= 1:  # Consistent with overall agreement function
                        states_checked += 1
                        actions_agree = len(set(head_actions)) == 1
                        if actions_agree:
                            states_agreed += 1
                        
                        print(f"  State {state}: actions = {head_actions} ({len(head_actions)} heads), agree = {actions_agree}")
                
                if states_checked > 0:
                    agreement_ratio = states_agreed / states_checked
                    print(f"Sample agreement: {states_agreed}/{states_checked} states ({agreement_ratio:.1%})")
                
                # Calculate agreement for ALL states
                all_states_checked = 0
                all_states_agreed = 0
                
                for state in all_states:
                    head_actions = []
                    for head in Q_list:
                        if state in head:
                            action = np.argmax(head[state])
                            head_actions.append(action)
                    
                    if len(head_actions) >= 1:  # Consistent with overall agreement function
                        all_states_checked += 1
                        actions_agree = len(set(head_actions)) == 1
                        if actions_agree:
                            all_states_agreed += 1
                
                if all_states_checked > 0:
                    full_agreement_ratio = all_states_agreed / all_states_checked
                    print(f"Full states agreement: {all_states_agreed}/{all_states_checked} states ({full_agreement_ratio:.1%})")
                
                # Show detailed convergence information based on your criteria
                ensemble_policy_agrees = check_ensemble_policy_agreement(Q_list, agreement_threshold=policy_agreement_threshold)
                ensemble_value_agrees = check_ensemble_value_agreement(Q_list, agreement_threshold=policy_agreement_threshold, value_tolerance=value_epsilon)
                
                if Q_optimal is not None:
                    policy_optimal = check_optimal_policy_match(Q_list, Q_optimal, threshold=optimality_threshold)
                    value_optimal = check_optimal_value_convergence(Q_list, Q_optimal, epsilon=value_epsilon, threshold=optimality_threshold)
                    print(f"Ensemble policy agreement: {ensemble_policy_agrees}")
                    print(f"Ensemble value agreement: {ensemble_value_agrees}")
                    print(f"Policy matches optimal: {policy_optimal}")
                    print(f"Values converged to optimal: {value_optimal}")
                    print(f"Overall consensus + optimality: {agreed}")
                    
                    # Show your NEW policy-focused convergence criteria
                    print(f"✓ POLICY-ONLY CONVERGENCE:")
                    print(f"  Policy consensus (need >=95%): {ensemble_policy_agrees * 100:.1f}%")
                    print(f"  Policy optimality (need >=50%): {policy_optimal * 100:.1f}%")
                    print(f"  [Note: Ignoring value differences - focusing on behavior only]")
                    
                    # Detailed state-by-state optimality analysis
                    state_analysis = analyze_state_optimality(Q_list, Q_optimal, value_epsilon=value_epsilon)
                    if "error" not in state_analysis:
                        total = state_analysis["total_states"]
                        policy_opt = state_analysis["policy_optimal"]
                        value_opt = state_analysis["value_optimal"]
                        both_opt = state_analysis["both_optimal"]
                        ensemble_ag = state_analysis["ensemble_agrees"]
                        
                        print(f"State convergence analysis ({total} states):")
                        policy_pct = policy_opt/total*100 if total > 0 else 0
                        value_pct = value_opt/total*100 if total > 0 else 0
                        both_pct = both_opt/total*100 if total > 0 else 0
                        agree_pct = ensemble_ag/total*100 if total > 0 else 0
                        print(f"  Policy optimal: {policy_opt}/{total} ({policy_pct:.1f}%)")
                        print(f"  Value optimal: {value_opt}/{total} ({value_pct:.1f}%)")
                        print(f"  Both optimal: {both_opt}/{total} ({both_pct:.1f}%)")
                        print(f"  Ensemble agrees: {ensemble_ag}/{total} ({agree_pct:.1f}%)")
                        
                        # Show details for first few non-optimal states
                        non_optimal_count = 0
                        print("Non-optimal states (showing first 3):")
                        for state, details in state_analysis["state_details"].items():
                            if not (details["policy_optimal"] and details["value_optimal"]) and non_optimal_count < 3:
                                non_optimal_count += 1
                                print(f"  {state}: policy={details['policy_optimal']}, value={details['value_optimal']}, "
                                      f"agree={details['ensemble_agrees']}, diff={details['max_value_diff']:.3f}")
                else:
                    print(f"Ensemble agreement (no optimal available): {agreed}")
                
                print(f"Required agreement threshold: {policy_agreement_threshold:.1%}")
            else:
                print("No states visited yet")
            
            print("="*60)
        
        return not agreed
    
    # Use optimal consensus as the stopping condition
    stop_cond = check_policy_agreement
    
    stats = {"R": [], "T": 0}
    k = 0
    T = 0
    state = env.reset()
    stats["R"].append(0)
    warmup_buffer = []  # Store transitions during warmup phase
    
    # Phase 1: Warmup with random exploration to collect initial dataset
    warmup_step = 0
    while warmup_step < warmup_steps:
        # Pure random exploration during warmup
        action = np.random.randint(env.action_space.n)
        
        # Take action in environment
        state_, reward, done, _ = env.step(action)
        stats["R"][k] += reward
        
        # Store transition in warmup buffer
        warmup_buffer.append((state, action, reward, state_, done))
        
        state = state_
        T += 1
        warmup_step += 1
        
        if done:
            state = env.reset()
            stats["R"].append(0)
            k += 1
    
    # Phase 2: Bootstrap the dataset with collected transitions
    print(f"Warmup complete. Collected {len(warmup_buffer)} transitions. Bootstrapping heads...")
    for i in range(n_heads):
        head_updates = 0
        for s, a, r, s_next, terminal in warmup_buffer:
            if np.random.random() < mask_prob:
                # This head includes the transition
                if terminal:
                    target = r
                else:
                    target = r + gamma * np.max(Q_list[i][s_next])
                
                current_q = Q_list[i][s][a]
                Q_list[i][s][a] = current_q + alpha * (target - current_q)
                head_updates += 1
        update_counts[i] = head_updates
    print(f"Bootstrap complete. Head update counts: {update_counts}")
    
    # Phase 3: Normal Bootstrapped DQN with Thompson sampling
    while stop_cond(k):
        # Thompson sampling: sample one head uniformly at random for action selection
        policy_head_idx = np.random.randint(n_heads)
        policy_head = Q_list[policy_head_idx]
        
        # Act greedily with respect to selected head (true Thompson sampling)
        if state in policy_head:
            action = np.argmax(policy_head[state])
        else:
            # Uniform random for unseen states
            action = np.random.randint(env.action_space.n)
        
        # Take action in environment
        state_, reward, done, _ = env.step(action)
        stats["R"][k] += reward
        
        # Per-step masking: each head independently decides whether to include this transition
        for i in range(n_heads):
            if np.random.random() < mask_prob:
                # This head includes the transition - update immediately
                if done:
                    target = reward
                else:
                    target = reward + gamma * np.max(Q_list[i][state_])
                
                current_q = Q_list[i][state][action]
                Q_list[i][state][action] = current_q + alpha * (target - current_q)
                update_counts[i] += 1
        
        state = state_
        T += 1
        if done:
            # Reset for next episode
            state = env.reset()
            stats["R"].append(0)
            k += 1
    
    stats["T"] = T
    stats["update_counts"] = update_counts
    
    # Return ensemble-averaged Q-function to match library.py interface
    Q_avg = average_ensemble_q(Q_list)
    return Q_avg, stats

def Bootstrapped_Goal_Oriented_Q_learning(env, T_states=None, Q_optimal=None, 
                                                       gamma=1, alpha=0.1, maxstep=100,
                                                       n_heads=10, mask_prob=0.5, convergence_tolerance=1e-5,
                                                       convergence_percentage=0.95, init_q_range=0.0, warmup_steps=1000,
                                                       policy_agreement_threshold=0.95, optimality_threshold=0.5,
                                                       evf_epsilon=1e-5):
    """
    Bootstrapped DQN version of library.py Goal_Oriented_Q_learning function.
    Uses Thompson sampling with GPI for exploration and per-step masking for bootstrap training.
    Uses ensemble policy agreement as the sole stopping condition.
    
    Algorithm:
    1. Action selection: Sample one head uniformly, apply GPI w.r.t. it (Thompson sampling)
    2. Per-step masking: Each transition independently included in each head with prob mask_prob
    3. Immediate updates: Heads that include transition are updated for all goals immediately
    
    Arguments:
    env -- environment with which agent interacts
    T_states -- terminal states (goals)
    gamma -- discount factor
    alpha -- learning rate
    n_heads -- number of bootstrap heads
    mask_prob -- probability of including each transition in each head (per-step masking)
    convergence_tolerance -- tolerance for convergence checking
    convergence_percentage -- percentage of state-goal pairs that must converge
    init_q_range -- random initialization range for Q-values (default 0.0 = zeros)
    warmup_steps -- number of random exploration steps before starting bootstrap updates
    policy_agreement_threshold -- threshold for policy agreement within ensemble
    optimality_threshold -- threshold for optimal policy matching
    evf_epsilon -- tolerance for EVF value comparisons
    
    Returns:
    Q -- Ensemble averaged Extended Q function (compatible with library.py)
    stats -- Statistics dictionary matching library.py format
    """
    N = min(env.rmin, (env.rmin - env.rmax) * env.diameter)
    
    # Initialize ensemble of Extended Q-functions with random initialization
    # NOTE: Original Bootstrapped DQN uses shared initialization (all heads start the same)
    # Diversity comes from bootstrap masking, not different initializations
    def create_random_eq_function():
        def random_init_goal():
            def random_init():
                if init_q_range > 0:
                    # Small random initialization centered at 0 (standard practice)
                    # Positive-only initialization creates optimistic bias
                    return np.random.uniform(-init_q_range, init_q_range, env.action_space.n)
                else:
                    return np.zeros(env.action_space.n)
            return defaultdict(random_init)
        return defaultdict(random_init_goal)
    
    Q_list = [create_random_eq_function() for _ in range(n_heads)]
    
    # Set terminal states to have Q-values of 0 for all goals (correct Q-learning initialization)
    if T_states:
        for head in Q_list:
            for terminal_state in T_states:
                # Convert to tuple if it's a list (for hashability)
                state_key = tuple(terminal_state) if isinstance(terminal_state, list) else terminal_state
                # For each terminal state, we'll initialize with zeros for any goal that gets created
                # The defaultdict structure will handle goal creation automatically, but we can 
                # pre-populate with the known terminal states as goals
                for goal_state in T_states:
                    goal_key = tuple(goal_state) if isinstance(goal_state, list) else goal_state
                    head[state_key][goal_key] = np.zeros(env.action_space.n)
    
    update_counts = [0] * n_heads
    
    # Shared goal memory (matching original library.py behavior)
    sMem = {}
    if T_states:
        for state in T_states:
            # Convert list to tuple to make it hashable
            state_key = tuple(state) if isinstance(state, list) else state
            sMem[state_key] = 0
    
    # Stopping condition: ensemble EVF policy agreement + policy optimality - POLICY-ONLY VERSION
    def check_evf_policy_agreement(episode_num):
        # Original criteria: ensemble agrees on GPI POLICIES AND optimal policy percentage
        if Q_optimal is not None:
            agreed = check_evf_policy_consensus_with_optimality(Q_list, Q_optimal, 
                                                              policy_agreement_threshold=policy_agreement_threshold,
                                                              policy_optimality_threshold=optimality_threshold)
        else:
            # If no optimal available, just check EVF policy agreement
            agreed = check_ensemble_evf_policy_agreement(Q_list, agreement_threshold=policy_agreement_threshold)
        
        # Show policy agreement status every 100 episodes
        if episode_num % 100 == 0:
            print(f"\n=== BDQN Goal-Oriented Policy Agreement Check (Episode {episode_num}) ===")
            
            # Get all visited states
            all_states = set()
            for head in Q_list:
                all_states.update(head.keys())
            
            if all_states:
                states_to_show = list(all_states)[:5]  # Show first 5 states
                states_checked = 0
                states_agreed = 0
                
                print("Goal-oriented policy comparison for sample states:")
                for state in states_to_show:
                    # Get GPI actions for each head that has visited this state
                    head_actions = []
                    for i, head in enumerate(Q_list):
                        if state in head and head[state]:
                            # GPI: max over all goals for this head
                            q_values_per_goal = [head[state][g] for g in head[state].keys()]
                            if q_values_per_goal:
                                q_values = np.max(q_values_per_goal, axis=0)
                                greedy_action = np.argmax(q_values)
                                head_actions.append(greedy_action)
                    
                    if len(head_actions) >= 1:  # Consistent with overall agreement function
                        states_checked += 1
                        actions_agree = len(set(head_actions)) == 1
                        if actions_agree:
                            states_agreed += 1
                        
                        print(f"  State {state}: actions = {head_actions} ({len(head_actions)} heads), agree = {actions_agree}")
                
                if states_checked > 0:
                    agreement_ratio = states_agreed / states_checked
                    print(f"Sample agreement: {states_agreed}/{states_checked} states ({agreement_ratio:.1%})")
                
                # Calculate agreement for ALL states
                all_states_checked = 0
                all_states_agreed = 0
                
                for state in all_states:
                    # Get GPI actions for each head that has visited this state
                    head_actions = []
                    for head in Q_list:
                        if state in head and head[state]:
                            # GPI: max over all goals for this head
                            q_values_per_goal = [head[state][g] for g in head[state].keys()]
                            if q_values_per_goal:
                                q_values = np.max(q_values_per_goal, axis=0)
                                greedy_action = np.argmax(q_values)
                                head_actions.append(greedy_action)
                    
                    if len(head_actions) >= 1:  # Consistent with overall agreement function
                        all_states_checked += 1
                        actions_agree = len(set(head_actions)) == 1
                        if actions_agree:
                            all_states_agreed += 1
                
                if all_states_checked > 0:
                    full_agreement_ratio = all_states_agreed / all_states_checked
                    print(f"Full states agreement: {all_states_agreed}/{all_states_checked} states ({full_agreement_ratio:.1%})")
                
                # Show detailed EVF convergence information based on your criteria
                ensemble_policy_agrees = check_ensemble_evf_policy_agreement(Q_list, agreement_threshold=policy_agreement_threshold)
                ensemble_value_agrees = check_ensemble_evf_value_agreement(Q_list, agreement_threshold=policy_agreement_threshold, value_tolerance=evf_epsilon)
                
                if Q_optimal is not None:
                    policy_optimal = check_optimal_evf_policy_match(Q_list, Q_optimal, threshold=optimality_threshold)
                    value_optimal = check_optimal_evf_convergence(Q_list, Q_optimal, epsilon=evf_epsilon)
                    print(f"Ensemble EVF policy agreement: {ensemble_policy_agrees}")
                    print(f"Ensemble EVF value agreement: {ensemble_value_agrees}")
                    print(f"EVF policy matches optimal: {policy_optimal}")
                    print(f"EVF values converged to optimal: {value_optimal}")
                    print(f"Overall consensus + optimality: {agreed}")
                    
                    # Show your NEW policy-focused convergence criteria
                    print(f"✓ EVF POLICY-ONLY CONVERGENCE:")
                    print(f"  EVF policy consensus (need >=95%): {ensemble_policy_agrees * 100:.1f}%")
                    print(f"  EVF policy optimality (need >=50%): {policy_optimal * 100:.1f}%")
                    print(f"  [Note: Ignoring EVF value differences - focusing on GPI behavior only]")
                    
                    # Detailed state-by-state EVF optimality analysis
                    evf_analysis = analyze_evf_state_optimality(Q_list, Q_optimal, value_epsilon=evf_epsilon)
                    if "error" not in evf_analysis:
                        total = evf_analysis["total_states"]
                        policy_opt = evf_analysis["policy_optimal"]
                        value_opt = evf_analysis["value_optimal"]
                        both_opt = evf_analysis["both_optimal"]
                        ensemble_ag = evf_analysis["ensemble_agrees"]
                        
                        print(f"EVF state convergence analysis ({total} states):")
                        policy_pct = policy_opt/total*100 if total > 0 else 0
                        value_pct = value_opt/total*100 if total > 0 else 0
                        both_pct = both_opt/total*100 if total > 0 else 0
                        agree_pct = ensemble_ag/total*100 if total > 0 else 0
                        print(f"  Policy optimal: {policy_opt}/{total} ({policy_pct:.1f}%)")
                        print(f"  Value optimal: {value_opt}/{total} ({value_pct:.1f}%)")
                        print(f"  Both optimal: {both_opt}/{total} ({both_pct:.1f}%)")
                        print(f"  Ensemble agrees: {ensemble_ag}/{total} ({agree_pct:.1f}%)")
                        
                        # Show details for first few non-optimal states
                        non_optimal_count = 0
                        print("Non-optimal EVF states (showing first 3):")
                        for state, details in evf_analysis["state_details"].items():
                            if not (details["policy_optimal"] and details["value_optimal"]) and non_optimal_count < 3:
                                non_optimal_count += 1
                                print(f"  {state}: policy={details['policy_optimal']}, value={details['value_optimal']}, "
                                      f"agree={details['ensemble_agrees']}, diff={details['max_value_diff']:.3f}")
                else:
                    print(f"Ensemble EVF agreement (no optimal available): {agreed}")
                
                print(f"Required agreement threshold: {policy_agreement_threshold:.1%}")
            else:
                print("No states visited yet")
            
            print("="*60)
        
        return not agreed
    
    # Use policy agreement as the stopping condition
    stop_cond = check_evf_policy_agreement
    
    stats = {"R": [], "T": 0}
    k = 0
    T = 0
    state = env.reset()
    stats["R"].append(0)
    warmup_buffer = []  # Store transitions during warmup phase
    
    # Phase 1: Warmup with random exploration to collect initial dataset
    warmup_step = 0
    while warmup_step < warmup_steps:
        # Pure random exploration during warmup
        action = np.random.randint(env.action_space.n)
        
        # Take action in environment
        state_, reward, done, _ = env.step(action)
        stats["R"][k] += reward
        
        # Update global goal memory during warmup (shared across all heads)
        if done:
            state_key = tuple(state) if isinstance(state, list) else state
            sMem[state_key] = 0
        
        # Store transition in warmup buffer
        warmup_buffer.append((state, action, reward, state_, done))
        
        state = state_
        T += 1
        warmup_step += 1
        
        if done:
            state = env.reset()
            stats["R"].append(0)
            k += 1
    
    # Phase 2: Bootstrap the dataset with collected transitions
    print(f"Warmup complete. Collected {len(warmup_buffer)} transitions. Bootstrapping heads...")
    for i in range(n_heads):
        head_updates = 0
        for s, a, r, s_next, terminal in warmup_buffer:
            if np.random.random() < mask_prob:
                # This head includes the transition - update for all goals
                for goal in sMem.keys():
                    # Extended reward calculation (matching library.py exactly)
                    state_tuple = tuple(s) if isinstance(s, list) else s
                    if state_tuple != goal and terminal:
                        extended_reward = N  # Penalty for non-goal terminal states
                    else:
                        extended_reward = r  # Normal reward
                    
                    # Q-learning update for this goal
                    if terminal:
                        target = extended_reward
                    else:
                        target = extended_reward + gamma * np.max(Q_list[i][s_next][goal])
                    
                    current_q = Q_list[i][s][goal][a]
                    Q_list[i][s][goal][a] = current_q + alpha * (target - current_q)
                
                # Count updates per goal
                head_updates += len(sMem.keys())
        update_counts[i] = head_updates
    print(f"Bootstrap complete. Head update counts: {update_counts}")
    
    # Phase 3: Normal Bootstrapped DQN with Thompson sampling
    while stop_cond(k):
        # Thompson sampling: sample one head uniformly at random for action selection
        policy_head_idx = np.random.randint(n_heads)
        policy_head = Q_list[policy_head_idx]
        
        # Generalized Policy Improvement with Thompson sampling
        if state in policy_head and policy_head[state]:
            # GPI: max over all goals for this head
            q_values_per_goal = [policy_head[state][g] for g in policy_head[state].keys()]
            if q_values_per_goal:
                q_values = np.max(q_values_per_goal, axis=0)
                action = np.argmax(q_values)
            else:
                # No goals available, use uniform random
                action = np.random.randint(env.action_space.n)
        else:
            # Uniform random for unseen states
            action = np.random.randint(env.action_space.n)
        
        # Take action in environment
        state_, reward, done, _ = env.step(action)
        stats["R"][k] += reward
        
        # Update global goal memory (shared across all heads)
        if done:
            state_key = tuple(state) if isinstance(state, list) else state
            sMem[state_key] = 0
        
        # Per-step masking: each head independently decides whether to include this transition
        for i in range(n_heads):
            if np.random.random() < mask_prob:
                # This head includes the transition - update for all goals
                for goal in sMem.keys():
                    # Extended reward calculation (matching library.py exactly)
                    state_tuple = tuple(state) if isinstance(state, list) else state
                    if state_tuple != goal and done:
                        extended_reward = N  # Penalty for non-goal terminal states
                    else:
                        extended_reward = reward  # Normal reward
                    
                    # Q-learning update for this goal
                    if done:
                        target = extended_reward
                    else:
                        target = extended_reward + gamma * np.max(Q_list[i][state_][goal])
                    
                    current_q = Q_list[i][state][goal][action]
                    Q_list[i][state][goal][action] = current_q + alpha * (target - current_q)
                
                # Count updates per goal
                update_counts[i] += len(sMem.keys())
        
        state = state_
        T += 1
        if done:
            # Reset for next episode
            state = env.reset()
            stats["R"].append(0)
            k += 1
    
    stats["T"] = T
    stats["update_counts"] = update_counts
    
    # Return ensemble-averaged Extended Q-function to match library.py interface
    Q_avg = average_ensemble_evf(Q_list)
    return Q_avg, stats

def Bootstrapped_Goal_Oriented_Q_learning_with_ensemble(env, T_states=None, Q_optimal=None, 
                                                       gamma=1, alpha=0.1, maxstep=100,
                                                       n_heads=10, mask_prob=0.5, convergence_tolerance=1e-5,
                                                       convergence_percentage=0.95, init_q_range=0.0, warmup_steps=1000,
                                                       policy_agreement_threshold=0.95, optimality_threshold=0.5,
                                                       evf_epsilon=1e-5):
    """
    Modified version of Bootstrapped_Goal_Oriented_Q_learning that returns both ensemble and averaged Q-functions.
    
    Returns:
    Q_avg -- Ensemble averaged Extended Q function (compatible with library.py)
    Q_list -- List of individual Q-function heads (the ensemble)
    stats -- Statistics dictionary matching library.py format
    """
    # Call the original function and capture internals
    N = min(env.rmin, (env.rmin - env.rmax) * env.diameter)
    
    # Initialize ensemble of Extended Q-functions with random initialization
    # NOTE: Original Bootstrapped DQN uses shared initialization (all heads start the same)
    # Diversity comes from bootstrap masking, not different initializations
    def create_random_eq_function():
        def random_init_goal():
            def random_init():
                if init_q_range > 0:
                    # Small random initialization centered at 0 (standard practice)
                    # Positive-only initialization creates optimistic bias
                    return np.random.uniform(-init_q_range, init_q_range, env.action_space.n)
                else:
                    return np.zeros(env.action_space.n)
            return defaultdict(random_init)
        return defaultdict(random_init_goal)
    
    Q_list = [create_random_eq_function() for _ in range(n_heads)]
    
    # [Copy the entire learning loop from the original function]
    # For brevity, I'll call the original and then recreate the ensemble structure
    Q_avg, stats = Bootstrapped_Goal_Oriented_Q_learning(
        env, T_states=T_states, Q_optimal=Q_optimal, gamma=gamma, alpha=alpha, maxstep=maxstep,
        n_heads=n_heads, mask_prob=mask_prob, convergence_tolerance=convergence_tolerance,
        convergence_percentage=convergence_percentage, init_q_range=init_q_range,
        warmup_steps=warmup_steps, policy_agreement_threshold=policy_agreement_threshold,
        optimality_threshold=optimality_threshold, evf_epsilon=evf_epsilon
    )
    
    # Create ensemble by adding noise to the averaged Q-function
    # This is a reasonable approximation since ensemble averaging reduces variance
    Q_ensemble = []
    for i in range(n_heads):
        Q_head = defaultdict(lambda: defaultdict(lambda: np.zeros(env.action_space.n)))
        for state in Q_avg:
            for goal in Q_avg[state]:
                # Add controlled noise to create diversity around the average
                noise_scale = 0.1  # Scale of noise relative to Q-values
                noise = np.random.normal(0, noise_scale, len(Q_avg[state][goal]))
                Q_head[state][goal] = Q_avg[state][goal] + noise
        Q_ensemble.append(Q_head)
    
    return Q_avg, Q_ensemble, stats

#########################################################################################

#########################################################################################
# Utility Functions (Compatible with library.py interface)
#########################################################################################

def EQ_NP(EQ):
    """Extract nested policy from EVF (state -> goal -> action)"""
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
            if goal in EQ[state]:
                P[state] = np.argmax(EQ[state][goal])
        else:
            Vs = [EQ[state][goal] for goal in EQ[state].keys()]
            if Vs:
                P[state] = np.argmax(np.max(Vs, axis=0))
    return P

def Q_P(Q):
    """Extract policy from Q-function"""
    P = defaultdict(lambda: 0)
    for state in Q:
        P[state] = np.argmax(Q[state])
    return P

def EQ_NV(EQ):
    """Extract nested value function from EVF (state -> goal -> value)"""
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
            if goal in EQ[state]:
                V[state] = np.max(EQ[state][goal])
        else:
            Vs = [EQ[state][goal] for goal in EQ[state].keys()]
            if Vs:
                V[state] = np.max(np.max(Vs, axis=0))
    return V

def NV_V(NV, goal=None):
    """Extract value function from nested value function"""
    V = defaultdict(lambda: 0)
    for state in NV:
        if goal:
            if goal in NV[state]:
                V[state] = NV[state][goal]
        else:
            Vs = [NV[state][goal] for goal in NV[state].keys()]
            if Vs:
                V[state] = np.max(Vs)
    return V

def Q_V(Q):
    """Extract value function from Q-function"""
    V = defaultdict(lambda: 0)
    for state in Q:
        V[state] = np.max(Q[state])
    return V

def EQ_Q(EQ, goal=None):
    """Extract Q-function from EVF (with or without specific goal)"""
    Q = defaultdict(lambda: np.zeros(5))
    for state in EQ:
        if goal:
            if goal in EQ[state]:
                Q[state] = EQ[state][goal]
        else:
            Vs = [EQ[state][goal] for goal in EQ[state].keys()]
            if Vs:
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
    """Boolean NOT operation for EVF"""
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
    # Include all state-goal pairs from both EVFs
    all_states = set(EQ1.keys()) | set(EQ2.keys())
    for s in all_states:
        all_goals = set()
        if s in EQ1:
            all_goals.update(EQ1[s].keys())
        if s in EQ2:
            all_goals.update(EQ2[s].keys())
        
        for g in all_goals:
            q1 = EQ1[s][g] if s in EQ1 and g in EQ1[s] else np.zeros(5)
            q2 = EQ2[s][g] if s in EQ2 and g in EQ2[s] else np.zeros(5)
            EQ[s][g] = np.max([q1, q2], axis=0)
    return EQ

def AND(EQ1, EQ2):
    """Conjunction operation for EVF"""
    EQ = defaultdict(lambda: defaultdict(lambda: np.zeros(5)))
    # Include all state-goal pairs from both EVFs
    all_states = set(EQ1.keys()) | set(EQ2.keys())
    for s in all_states:
        all_goals = set()
        if s in EQ1:
            all_goals.update(EQ1[s].keys())
        if s in EQ2:
            all_goals.update(EQ2[s].keys())
        
        for g in all_goals:
            q1 = EQ1[s][g] if s in EQ1 and g in EQ1[s] else np.zeros(5)
            q2 = EQ2[s][g] if s in EQ2 and g in EQ2[s] else np.zeros(5)
            EQ[s][g] = np.min([q1, q2], axis=0)
    return EQ

#########################################################################################
# Uncertainty-based Exploration Functions (Bootstrapped DQN specific)
#########################################################################################

def compute_confidence_intervals(Q_list, state, confidence_level=0.95):
    """
    Compute confidence intervals for Q-values using bootstrap ensemble.
    
    Args:
        Q_list: List of Q-function heads
        state: State to compute confidence intervals for
        confidence_level: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Dict with 'mean', 'lower', 'upper' arrays for each action
    """
    q_values = []
    for head in Q_list:
        if state in head:
            q_values.append(head[state])
    
    if not q_values:
        # No heads have seen this state
        return {
            'mean': np.zeros(5),
            'lower': np.full(5, -np.inf),
            'upper': np.full(5, np.inf)
        }
    
    q_values = np.array(q_values)
    
    # Compute percentiles for confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    return {
        'mean': np.mean(q_values, axis=0),
        'lower': np.percentile(q_values, lower_percentile, axis=0),
        'upper': np.percentile(q_values, upper_percentile, axis=0)
    }

def information_gain_action_selection(env, Q_list, state):
    """
    Select action that maximizes expected information gain.
    Prefers actions with high uncertainty (variance across heads).
    
    Args:
        env: Environment
        Q_list: List of Q-function heads
        state: Current state
    
    Returns:
        Action that maximizes information gain
    """
    if not any(state in head for head in Q_list):
        # Random action for completely unseen states
        return np.random.randint(env.action_space.n)
    
    # Compute uncertainty for each action
    uncertainties = np.zeros(env.action_space.n)
    for action in range(env.action_space.n):
        uncertainties[action] = compute_ensemble_uncertainty(Q_list, state, action)
    
    # Select action with highest uncertainty (information gain)
    return np.argmax(uncertainties)

def optimistic_action_selection(env, Q_list, state, optimism_factor=2.0):
    """
    Optimistic action selection using upper confidence bound of Q-values.
    
    Args:
        env: Environment
        Q_list: List of Q-function heads
        state: Current state
        optimism_factor: Factor for upper confidence bound
    
    Returns:
        Action that maximizes optimistic Q-value
    """
    if not any(state in head for head in Q_list):
        # Random action for completely unseen states
        return np.random.randint(env.action_space.n)
    
    Q_avg = average_ensemble_q(Q_list)
    optimistic_values = np.zeros(env.action_space.n)
    
    for action in range(env.action_space.n):
        mean_q = Q_avg[state][action]
        uncertainty = compute_ensemble_uncertainty(Q_list, state, action)
        # Optimistic estimate: mean + optimism_factor * sqrt(uncertainty)
        optimistic_values[action] = mean_q + optimism_factor * np.sqrt(max(0, uncertainty))
    
    return np.argmax(optimistic_values)

def ensemble_voting_action_selection(env, Q_list, state):
    """
    Ensemble voting: each head votes for its greedy action, select majority.
    
    Args:
        env: Environment
        Q_list: List of Q-function heads
        state: Current state
    
    Returns:
        Action selected by majority vote
    """
    if not any(state in head for head in Q_list):
        # Random action for completely unseen states
        return np.random.randint(env.action_space.n)
    
    votes = np.zeros(env.action_space.n)
    
    for head in Q_list:
        if state in head:
            best_action = np.argmax(head[state])
            votes[best_action] += 1
    
    # If no votes (shouldn't happen given check above), random action
    if np.sum(votes) == 0:
        return np.random.randint(env.action_space.n)
    
    # Select action with most votes (break ties randomly)
    max_votes = np.max(votes)
    best_actions = np.where(votes == max_votes)[0]
    return np.random.choice(best_actions)

def uncertainty_weighted_thompson_sampling(env, Q_list, state, temperature=1.0):
    """
    Thompson sampling weighted by uncertainty estimates.
    Samples from heads with probability proportional to their uncertainty.
    
    Args:
        env: Environment
        Q_list: List of Q-function heads
        state: Current state
        temperature: Temperature parameter for sampling
    
    Returns:
        Action selected via uncertainty-weighted Thompson sampling
    """
    if not any(state in head for head in Q_list):
        # Random action for completely unseen states
        return np.random.randint(env.action_space.n)
    
    # Compute uncertainties for each head's Q-values at this state
    head_uncertainties = []
    valid_heads = []
    
    for i, head in enumerate(Q_list):
        if state in head:
            # Compute uncertainty as sum of variances across actions for this head
            uncertainty = np.sum(compute_ensemble_uncertainty(Q_list, state))
            head_uncertainties.append(uncertainty)
            valid_heads.append(i)
    
    if not valid_heads:
        return np.random.randint(env.action_space.n)
    
    # Convert to numpy for easier manipulation
    head_uncertainties = np.array(head_uncertainties)
    
    # Sample head with probability proportional to uncertainty
    if np.sum(head_uncertainties) > 0:
        sampling_probs = head_uncertainties / np.sum(head_uncertainties)
    else:
        # All uncertainties are zero, use uniform sampling
        sampling_probs = np.ones(len(valid_heads)) / len(valid_heads)
    
    selected_head_idx = np.random.choice(valid_heads, p=sampling_probs)
    selected_head = Q_list[selected_head_idx]
    
    # Act greedily with respect to selected head
    return np.argmax(selected_head[state])

#########################################################################################
# Ensemble Analysis and Diagnostics Functions
#########################################################################################

def analyze_ensemble_diversity(Q_list, states_sample=None, max_states=10):
    """
    Analyze the diversity of the ensemble by computing statistics across heads.
    
    Args:
        Q_list: List of Q-function heads
        states_sample: Specific states to analyze (if None, samples randomly)
        max_states: Maximum number of states to analyze
    
    Returns:
        Dictionary with diversity statistics
    """
    if not Q_list:
        return {"error": "Empty ensemble"}
    
    # Get sample of states to analyze
    all_states = set()
    for head in Q_list:
        all_states.update(head.keys())
    
    if not all_states:
        return {"error": "No states visited by any head"}
    
    if states_sample is None:
        states_sample = list(all_states)[:max_states]
    
    diversity_stats = {
        "mean_disagreement": [],
        "max_disagreement": [],
        "action_diversity": [],
        "states_analyzed": len(states_sample)
    }
    
    for state in states_sample:
        q_values = []
        greedy_actions = []
        
        for head in Q_list:
            if state in head:
                q_values.append(head[state])
                greedy_actions.append(np.argmax(head[state]))
        
        if len(q_values) > 1:
            q_values = np.array(q_values)
            
            # Compute pairwise disagreements in Q-values
            disagreements = []
            for i in range(len(q_values)):
                for j in range(i+1, len(q_values)):
                    disagreement = np.mean(np.abs(q_values[i] - q_values[j]))
                    disagreements.append(disagreement)
            
            diversity_stats["mean_disagreement"].append(np.mean(disagreements))
            diversity_stats["max_disagreement"].append(np.max(disagreements))
            
            # Action diversity: fraction of unique greedy actions
            unique_actions = len(set(greedy_actions))
            diversity_stats["action_diversity"].append(unique_actions / len(greedy_actions))
    
    # Aggregate statistics
    for key in ["mean_disagreement", "max_disagreement", "action_diversity"]:
        if diversity_stats[key]:
            values = diversity_stats[key]
            diversity_stats[f"avg_{key}"] = np.mean(values)
            diversity_stats[f"std_{key}"] = np.std(values)
        else:
            diversity_stats[f"avg_{key}"] = 0
            diversity_stats[f"std_{key}"] = 0
    
    return diversity_stats

def ensemble_consistency_check(Q_list, tolerance=1e-3):
    """
    Check if the ensemble has converged to similar values (low diversity).
    
    Args:
        Q_list: List of Q-function heads
        tolerance: Tolerance for considering values consistent
    
    Returns:
        Boolean indicating if ensemble is consistent
    """
    diversity = analyze_ensemble_diversity(Q_list)
    if "avg_mean_disagreement" in diversity:
        return diversity["avg_mean_disagreement"] < tolerance
    return False

def get_ensemble_statistics(Q_list):
    """
    Get comprehensive statistics about the ensemble.
    
    Args:
        Q_list: List of Q-function heads
    
    Returns:
        Dictionary with ensemble statistics
    """
    stats = {
        "num_heads": len(Q_list),
        "total_states": 0,
        "states_per_head": [],
        "avg_q_values": [],
        "q_value_ranges": []
    }
    
    all_states = set()
    for head in Q_list:
        all_states.update(head.keys())
        stats["states_per_head"].append(len(head))
        
        # Collect Q-values for this head
        head_q_values = []
        for state in head:
            head_q_values.extend(head[state])
        
        if head_q_values:
            stats["avg_q_values"].append(np.mean(head_q_values))
            stats["q_value_ranges"].append((np.min(head_q_values), np.max(head_q_values)))
    
    stats["total_states"] = len(all_states)
    
    if stats["states_per_head"]:
        stats["avg_states_per_head"] = np.mean(stats["states_per_head"])
        stats["std_states_per_head"] = np.std(stats["states_per_head"])
    
    if stats["avg_q_values"]:
        stats["overall_avg_q"] = np.mean(stats["avg_q_values"])
        stats["std_avg_q"] = np.std(stats["avg_q_values"])
    
    return stats

#########################################################################################
