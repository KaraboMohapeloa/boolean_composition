import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MNFDQNQNetwork(nn.Module):
    """
    Multiplicative Normalizing Flows DQN Q-network for exploration.
    This network outputs a distribution over Q-values for each action.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=8, n_flows=2):
        super(MNFDQNQNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_flows = n_flows
        # Base MLP
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_q = nn.Linear(hidden_dim, action_dim)
        # MNF parameters (simple version: multiplicative noise)
        self.log_var = nn.Parameter(torch.zeros(action_dim))
        self.mu = nn.Parameter(torch.zeros(action_dim))
        # For more complex flows, add flow layers here

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q = self.fc_q(x)
        return q

    def sample_q(self, state):
        """
        Sample Q-values from the posterior using multiplicative normalizing flows.
        For simplicity, we use Gaussian noise here. Replace with MNF for full implementation.
        """
        q = self.forward(state)
        noise = torch.randn_like(q) * torch.exp(0.5 * self.log_var) + self.mu
        q_sample = q * noise
        return q_sample.detach().cpu().numpy()
