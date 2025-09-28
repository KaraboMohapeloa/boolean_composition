import numpy as np
import ast

# Tabular Bayesian Q-table for small state spaces
class BayesianQTable:
    def __init__(self, n_states, n_actions, prior_mean=0.0, prior_var=1.0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.mean = np.ones((n_states, n_actions)) * prior_mean
        self.var = np.ones((n_states, n_actions)) * prior_var
        self.count = np.zeros((n_states, n_actions))

    def sample_q(self, state):
        # Sample Q-values for all actions from the posterior
        return np.random.normal(self.mean[state], np.sqrt(self.var[state]))

    def update(self, state, action, target, alpha=0.1):
        # Bayesian update: simple running mean/variance (could use more advanced update)
        self.count[state, action] += 1
        old_mean = self.mean[state, action]
        # Update mean
        self.mean[state, action] += alpha * (target - self.mean[state, action])
        # Update variance (Welford's online algorithm)
        if self.count[state, action] > 1:
            delta = target - old_mean
            self.var[state, action] += (delta * (target - self.mean[state, action]) - self.var[state, action]) / self.count[state, action]
        else:
            self.var[state, action] = 1.0  # Reset to prior for first update

def to_one_hot(state, state_dim):
    """
    Convert a discrete state index or (x, y) tuple to a one-hot numpy array.
    If state is a tuple, map it to its index in possiblePositions.
    """
    def get_index(state, possiblePositions):
        # Recursively extract position tuple from nested lists
        if isinstance(state, list):
            # If the last element is a tuple, use it
            if len(state) > 0 and isinstance(state[-1], tuple):
                state = state[-1]
            # If the last element is a list, recurse
            elif len(state) > 0 and isinstance(state[-1], list):
                return get_index(state[-1], possiblePositions)
        if isinstance(state, (tuple, list)) and not isinstance(state, str):
            if possiblePositions is not None:
                try:
                    return possiblePositions.index(tuple(state))
                except Exception:
                    pass
        if isinstance(state, str) and state.isdigit():
            return int(state)
        if isinstance(state, (np.integer, np.floating)):
            return int(state)
        raise ValueError(f"Cannot convert state {state} to index for one-hot encoding.")

    one_hot = np.zeros(state_dim, dtype=np.float32)
    idx = get_index(state, to_one_hot.possiblePositions if hasattr(to_one_hot, 'possiblePositions') else None)
    one_hot[idx] = 1.0
    return one_hot


#########################################################################################
def mnfdqn_generalised_policy_improvement(env, Q, method="sample"):
    """
    Generalized policy improvement for MNF-DQN with multi-goal Q-networks.

    Arguments:
    env -- environment with which agent interacts
    Q -- MNF-DQN Q-network with a .sample_q(state, goal) or .sample_q(state) method
    method -- 'sample' (default): sample Q-values and act greedily

    Returns:
    policy_improved -- Improved policy function (state, goal) -> action probabilities
    """
    state_dim = env.observation_space.n
    possiblePositions = env.possiblePositions if hasattr(env, 'possiblePositions') else None
    def policy_improved(state, goal=None):
        n_actions = env.action_space.n
        to_one_hot.possiblePositions = possiblePositions
        # Parse state string to list if needed
        if isinstance(state, str):
            try:
                state = ast.literal_eval(state)
            except Exception:
                pass
        state_vec = to_one_hot(state, state_dim)
        # For tabular, state_vec is one-hot, so get index
        idx = np.argmax(state_vec)
        if method == "sample":
            q_sample = Q.sample_q(idx)
            best_action = np.random.choice(np.flatnonzero(q_sample == q_sample.max()))
            probs = np.zeros(n_actions)
            probs[best_action] = 1.0
            return probs
        else:
            raise ValueError(f"Unknown method: {method}")
    return policy_improved

#########################################################################################

def mnfdqn_policy_improvement(env, Q, method="sample"):
    """
    Policy improvement for MNF-DQN: sample Q-values from the MNF posterior and act greedily.
    Arguments:
    env -- environment with which agent interacts
    Q -- MNF-DQN Q-network with a .sample_q(state) method
    method -- 'sample' (default): sample Q-values and act greedily
    Returns:
    policy_improved -- function(state) -> action probabilities
    """
    state_dim = env.observation_space.n
    possiblePositions = env.possiblePositions if hasattr(env, 'possiblePositions') else None
    def policy_improved(state):
        n_actions = env.action_space.n
        to_one_hot.possiblePositions = possiblePositions
        # Parse state string to list if needed
        if isinstance(state, str):
            try:
                state = ast.literal_eval(state)
            except Exception:
                pass
        state_vec = to_one_hot(state, state_dim)
        idx = np.argmax(state_vec)
        if method == "sample":
            q_sample = Q.sample_q(idx)
            best_action = np.random.choice(np.flatnonzero(q_sample == q_sample.max()))
            probs = np.zeros(n_actions)
            probs[best_action] = 1.0
            return probs
        else:
            raise ValueError(f"Unknown method: {method}")
    return policy_improved

#########################################################################################


    


def Q_equal(Q1,Q2,epsilon=1e-5):    
    for state in Q1:
        for action in range(len(Q1[state])): 
            v1 = Q1[state][action]
            v2 = Q2[state][action]
            if abs(v1-v2)>epsilon:
                return False
    return True

def EQ_equal(EQ1,EQ2,epsilon=1e-5):    
    for state in EQ1:
        for goal in EQ1[state]:
            for action in range(len(EQ1[state][goal])): 
                v1 = EQ1[state][goal][action]
                v2 = EQ2[state][goal][action]
                if not (abs(v1-v2)<epsilon or (v1<-30 and v2<-30)):
                    return False
    return True


#########################################################################################
def Q_learning(env, Q_optimal=None, gamma=1, alpha=1, maxiter=100, maxstep=100):
    """
    Implements Q_learning using MNFDQNQNetwork

    Arguments:
    env -- environment with which agent interacts
    gamma -- discount factor
    alpha -- learning rate
    maxiter -- maximum number of episodes

    Returns:
    Q -- Trained MNFDQNQNetwork instance
    """
    # For Discrete observation space, use n as state_dim
    Q = BayesianQTable(env.observation_space.n, env.action_space.n)
    behaviour_policy = mnfdqn_policy_improvement(env, Q, method="sample")

    stop_cond = lambda k: k < maxiter

    stats = {"R":[], "T":0}
    k=0
    T=0
    state = env.reset()
    stats["R"].append(0)
    while stop_cond(k):
        probs = behaviour_policy(state)
        action = np.random.choice(np.arange(len(probs)), p=probs)
        state_, reward, done, _ = env.step(action)
        stats["R"][k] += reward
        # Parse state string to list if needed
        s = state
        s_ = state_
        if isinstance(s, str):
            try:
                s = ast.literal_eval(s)
            except Exception:
                pass
        if isinstance(s_, str):
            try:
                s_ = ast.literal_eval(s_)
            except Exception:
                pass
        # Convert to index
        to_one_hot.possiblePositions = env.possiblePositions if hasattr(env, 'possiblePositions') else None
        state_dim = env.observation_space.n
        idx = np.argmax(to_one_hot(s, state_dim))
        idx_ = np.argmax(to_one_hot(s_, state_dim))
        q_next = Q.sample_q(idx_)
        G = 0 if done else np.max(q_next)
        TD_target = reward + gamma*G
        Q.update(idx, action, TD_target, alpha)
        state = state_
        T+=1
        if done:
            state = env.reset()
            stats["R"].append(0)
            k+=1
    stats["T"] = T
    return Q, stats

def Goal_Oriented_Q_learning(env, T_states=None, Q_optimal=None, gamma=1, alpha=1, maxiter=100, maxstep=100):
    """
    Implements Goal Oriented Q-learning using MNFDQNQNetwork
            Q = MNFDQNQNetwork(env.observation_space.n, env.action_space.n)
            behaviour_policy = mnfdqn_policy_improvement(env, Q, method="sample")
    env -- environment with which agent interacts
    gamma -- discount factor
    alpha -- learning rate
    maxiter -- maximum number of episodes

    Returns:
    Q -- Trained MNFDQNQNetwork instance (multi-goal)
    stats -- Training statistics
    """
    N = min(env.rmin, (env.rmin - env.rmax) * env.diameter)
    Q = BayesianQTable(env.observation_space.n, env.action_space.n)
    behaviour_policy = mnfdqn_generalised_policy_improvement(env, Q, method="sample")

    sMem = {}
    stop_cond = lambda k: k < maxiter
    stats = {"R": [], "T": 0}
    k = 0
    T = 0
    state = env.reset()
    stats["R"].append(0)
    while stop_cond(k):
        probs = behaviour_policy(state, None)
        action = np.random.choice(np.arange(len(probs)), p=probs)
        state_, reward, done, _ = env.step(action)
        stats["R"][k] += reward
        if done:
            sMem[state] = 0
        for goal in sMem.keys():
            if state != goal and done:
                reward_ = N
            else:
                reward_ = reward
            # Parse state string to list if needed
            s = state
            s_ = state_
            if isinstance(s, str):
                try:
                    s = ast.literal_eval(s)
                except Exception:
                    pass
            if isinstance(s_, str):
                try:
                    s_ = ast.literal_eval(s_)
                except Exception:
                    pass
            # Convert to index
            to_one_hot.possiblePositions = env.possiblePositions if hasattr(env, 'possiblePositions') else None
            state_dim = env.observation_space.n
            idx = np.argmax(to_one_hot(s, state_dim))
            idx_ = np.argmax(to_one_hot(s_, state_dim))
            q_next = Q.sample_q(idx_)
            G = 0 if done else np.max(q_next)
            TD_target = reward_ + gamma * G
            Q.update(idx, action, TD_target, alpha)
        state = state_
        T += 1
        if done:
            state = env.reset()
            stats["R"].append(0)
            k += 1
    stats["T"] = T
    return Q, stats



#########################################################################################
