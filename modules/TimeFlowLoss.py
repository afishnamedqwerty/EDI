import torch
import torch.nn as nn
from torch.distributions import Normal

class TimeFlowLoss(nn.Module):
    def __init__(self, n_flows=5, hidden_dim=128):
        super().__init__()
        self.n_flows = n_flows  # Number of flow transformations
        self.hidden_dim = hidden_dim
        
        # Flow parameters: learn affine transformations for each flow step
        self.flow_weights = nn.Parameter(torch.randn(n_flows, hidden_dim))
        self.flow_biases = nn.Parameter(torch.randn(n_flows, hidden_dim))
        
        # Final distribution parameters (e.g., mean and std for a Gaussian)
        self.mean = nn.Parameter(torch.zeros(hidden_dim))
        self.log_std = nn.Parameter(torch.zeros(hidden_dim))
    
    def forward(self, x):
        """
        Forward pass of the TimeFlow Loss.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            loss: Scalar tensor representing the loss
        """
        batch_size, seq_len, d_model = x.size()
        
        # Reshape for processing flows
        x = x.view(-1, d_model)  # [batch_size * seq_len, d_model]
        
        # Apply flow transformations
        for i in range(self.n_flows):
            w = self.flow_weights[i]
            b = self.flow_biases[i]
            
            # Affine transformation: x -> (x * w + b)
            x = x * w + b
            
            # Invertibility check and numerical stability
            if i < self.n_flows - 1:
                # Apply sigmoid to maintain monotonicity and invertibility
                x = torch.sigmoid(x)
        
        # Compute the final distribution parameters
        mu = self.mean.unsqueeze(0)  # [1, d_model]
        std = torch.exp(self.log_std.unsqueeze(0)) + 1e-6  # [1, d_model]
        
        # Compute log-likelihood under the target Gaussian distribution
        dist = Normal(mu, std)
        log_prob = dist.log_prob(x)
        
        # Sum over features and sequence length
        loss = -torch.mean(torch.sum(log_prob, dim=1))
        
        return loss
