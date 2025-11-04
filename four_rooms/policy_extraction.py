import numpy as np
from collections import defaultdict, Counter
import deepdish as dd
from bdqn_library import EQ_P

def extract_head_policy(Q_head, goal=None):
    """
    Extract policy from a single Q-function head using GPI (Generalized Policy Improvement).
    
    Args:
        Q_head: Single Q-function head (Extended Value Function format)
        goal: Specific goal to extract policy for (None = use all goals with GPI)
    
    Returns:
        Dictionary mapping state -> action for optimal policy
    """
    return EQ_P(Q_head, goal)

def compute_majority_vote_policy(Q_ensemble, goal=None):
    """
    Compute majority vote policy across all heads in the ensemble.
    
    Args:
        Q_ensemble: List of Q-function heads
        goal: Specific goal to extract policy for (None = use all goals with GPI)
    
    Returns:
        Dictionary mapping state -> action for majority vote policy
        Dictionary mapping state -> vote_counts for debugging
    """
    if not Q_ensemble:
        return {}, {}
    
    # Extract policy from each head
    head_policies = []
    for head in Q_ensemble:
        policy = extract_head_policy(head, goal)
        head_policies.append(policy)
    
    # Find all states that appear in any head
    all_states = set()
    for policy in head_policies:
        all_states.update(policy.keys())
    
    # Compute majority vote for each state
    majority_policy = {}
    vote_counts = {}
    
    for state in all_states:
        # Collect votes from all heads for this state
        votes = []
        for policy in head_policies:
            if state in policy:
                votes.append(policy[state])
        
        if votes:
            # Count votes for each action
            vote_counter = Counter(votes)
            # Choose action with most votes (ties broken arbitrarily)
            majority_action = vote_counter.most_common(1)[0][0]
            majority_policy[state] = majority_action
            vote_counts[state] = dict(vote_counter)
        else:
            # No head has seen this state - use default action (0)
            majority_policy[state] = 0
            vote_counts[state] = {0: 0}  # No votes
    
    return majority_policy, vote_counts

def extract_ensemble_policies_for_task(ensemble_data, task_idx, task_goals):
    """
    Extract majority vote policies for a specific task from ensemble data.
    
    Args:
        ensemble_data: Ensemble data saved from exp3_bdqn_save_ensemble.py
        task_idx: Index of the task in the Tasks list
        task_goals: Goal positions for this task
    
    Returns:
        Dictionary containing majority vote policies and vote statistics
    """
    # Map task indices to the component policies
    # Tasks = [
    #     [],[(3,3),(3,9),(9,3),(9,9)],[(3,3)],[(3,9)],[(9,3)],[(9,9)],
    #     [(3,3),(3,9)],[(9,3),(9,9)],[(3,3),(9,3)],[(3,9),(9,9)],
    #     [(3,3),(3,9),(9,3)],[(3,3),(3,9),(9,9)],[(3,3),(9,3),(9,9)],
    #     [(3,9),(9,3),(9,9)],[(3,3),(9,9)],[(3,9),(9,3)]
    # ]
    
    result = {
        'task_idx': task_idx,
        'task_goals': task_goals,
        'majority_policy': {},
        'vote_counts': {},
        'policy_source': ''
    }
    
    # For base tasks that were directly learned, use their ensembles
    if task_idx == 6:  # A = [(3,3),(3,9)]
        if 'A_ensemble' in ensemble_data and ensemble_data['A_ensemble']:
            policy, votes = compute_majority_vote_policy(ensemble_data['A_ensemble'])
            result['majority_policy'] = policy
            result['vote_counts'] = votes
            result['policy_source'] = 'A_ensemble'
        else:
            print(f"Warning: A_ensemble not found for task {task_idx}")
    
    elif task_idx == 8:  # B = [(3,3),(9,3)]
        if 'B_ensemble' in ensemble_data and ensemble_data['B_ensemble']:
            policy, votes = compute_majority_vote_policy(ensemble_data['B_ensemble'])
            result['majority_policy'] = policy
            result['vote_counts'] = votes
            result['policy_source'] = 'B_ensemble'
        else:
            print(f"Warning: B_ensemble not found for task {task_idx}")
    
    elif task_idx == 1:  # EQ_max = [(3,3),(3,9),(9,3),(9,9)]
        if 'EQ_max_ensemble' in ensemble_data and ensemble_data['EQ_max_ensemble']:
            policy, votes = compute_majority_vote_policy(ensemble_data['EQ_max_ensemble'])
            result['majority_policy'] = policy
            result['vote_counts'] = votes
            result['policy_source'] = 'EQ_max_ensemble'
        else:
            print(f"Warning: EQ_max_ensemble not found for task {task_idx}")
    
    elif task_idx == 0:  # EQ_min = []
        if 'EQ_min_ensemble' in ensemble_data and ensemble_data['EQ_min_ensemble']:
            policy, votes = compute_majority_vote_policy(ensemble_data['EQ_min_ensemble'])
            result['majority_policy'] = policy
            result['vote_counts'] = votes
            result['policy_source'] = 'EQ_min_ensemble'
        else:
            print(f"Warning: EQ_min_ensemble not found for task {task_idx}")
    
    else:
        # For composed tasks, we would need to compose the base ensembles
        # This is more complex as it involves logical operations on ensembles
        # For now, return empty policy with a note
        result['policy_source'] = f'composed_task_{task_idx}_not_implemented'
        print(f"Warning: Composed task {task_idx} policy extraction not implemented yet")
    
    return result

def load_and_extract_policies(env_type, target_task_indices=None):
    """
    Load ensemble data and extract majority vote policies for specified tasks.
    
    Args:
        env_type: Environment type (0-3)
        target_task_indices: List of task indices to extract (None = all available)
    
    Returns:
        Dictionary mapping task_idx -> policy extraction result
    """
    try:
        ensemble_fname = f"exps_data/exp3_bdqn_ensembles_{env_type}.h5"
        ensemble_data = dd.io.load(ensemble_fname)
        print(f"Loaded ensemble data from {ensemble_fname}")
    except FileNotFoundError:
        print(f"Error: {ensemble_fname} not found. Run exp3_bdqn_save_ensemble.py first.")
        return {}
    
    # Task definitions for reference
    Tasks = [
        [],[(3,3),(3,9),(9,3),(9,9)],[(3,3)],[(3,9)],[(9,3)],[(9,9)],
        [(3,3),(3,9)],[(9,3),(9,9)],[(3,3),(9,3)],[(3,9),(9,9)],
        [(3,3),(3,9),(9,3)],[(3,3),(3,9),(9,9)],[(3,3),(9,3),(9,9)],
        [(3,9),(9,3),(9,9)],[(3,3),(9,9)],[(3,9),(9,3)]
    ]
    
    # Default to base tasks that we can extract directly
    if target_task_indices is None:
        target_task_indices = [0, 1, 6, 8]  # EQ_min, EQ_max, A, B
    
    extracted_policies = {}
    
    for task_idx in target_task_indices:
        if task_idx < len(Tasks):
            task_goals = Tasks[task_idx]
            result = extract_ensemble_policies_for_task(ensemble_data, task_idx, task_goals)
            extracted_policies[task_idx] = result
            
            # Print summary
            if result['majority_policy']:
                policy_size = len(result['majority_policy'])
                print(f"Task {task_idx} ({task_goals}): Extracted policy for {policy_size} states from {result['policy_source']}")
            else:
                print(f"Task {task_idx} ({task_goals}): No policy extracted - {result['policy_source']}")
        else:
            print(f"Warning: Task index {task_idx} out of range")
    
    return extracted_policies

def analyze_policy_agreement(vote_counts, min_agreement=0.8):
    """
    Analyze the level of agreement in majority vote policies.
    
    Args:
        vote_counts: Dictionary mapping state -> vote counts
        min_agreement: Minimum agreement threshold to consider "strong agreement"
    
    Returns:
        Dictionary with agreement statistics
    """
    agreement_stats = {
        'total_states': len(vote_counts),
        'strong_agreement_states': 0,
        'weak_agreement_states': 0,
        'disagreement_states': 0,
        'average_agreement': 0.0,
        'agreement_per_state': {}
    }
    
    total_agreement = 0.0
    
    for state, votes in vote_counts.items():
        if not votes:
            agreement_stats['agreement_per_state'][state] = 0.0
            continue
        
        total_votes = sum(votes.values())
        max_votes = max(votes.values())
        agreement_ratio = max_votes / total_votes if total_votes > 0 else 0.0
        
        agreement_stats['agreement_per_state'][state] = agreement_ratio
        total_agreement += agreement_ratio
        
        if agreement_ratio >= min_agreement:
            agreement_stats['strong_agreement_states'] += 1
        elif agreement_ratio >= 0.5:
            agreement_stats['weak_agreement_states'] += 1
        else:
            agreement_stats['disagreement_states'] += 1
    
    if agreement_stats['total_states'] > 0:
        agreement_stats['average_agreement'] = total_agreement / agreement_stats['total_states']
    
    return agreement_stats

# Example usage and testing functions
def test_policy_extraction():
    """Test policy extraction functionality."""
    print("Testing policy extraction functionality...")
    
    # Test for all environment types
    for env_type in range(4):
        print(f"\n=== Environment Type {env_type} ===")
        
        # Extract policies for base tasks
        policies = load_and_extract_policies(env_type, target_task_indices=[0, 1, 6, 8])
        
        for task_idx, result in policies.items():
            if result['vote_counts']:
                agreement_stats = analyze_policy_agreement(result['vote_counts'])
                print(f"  Task {task_idx}: {agreement_stats['strong_agreement_states']}/{agreement_stats['total_states']} states with strong agreement (avg: {agreement_stats['average_agreement']:.3f})")

if __name__ == "__main__":
    test_policy_extraction()