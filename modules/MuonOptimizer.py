import torch
from torch.optim import Optimizer

class MuonOptimizer(Optimizer):
    """
    Implementation of the Muon optimizer for training large language models.
    
    References:
        - Bernstein, E., & Newhouse, J. (2024). A specialized steepest descent method for pretraining LLMs.
        - Li, J., & Hong, M. (2025). A Note on the Convergence of Muon and Further.
    """
    
    def __init__(self, params, lr=0.001, mu=0.99, spectral_norm=True):
        """
        Initialize the Muon optimizer with given parameters and hyperparameters.
        
        Args:
            params (iterable): iterable of parameters to optimize
            lr (float): learning rate
            mu (float): momentum coefficient
            spectral_norm (bool): whether to apply spectral normalization
        """
        super().__init__(params)
        self.lr = lr
        self.mu = mu
        self.spectral_norm = spectral_norm
        # Momentum buffer to store the combined gradient and momentum terms
        self.momentum_buffer = {}
        
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss. Default: None
        """
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Compute gradient and update momentum
                grad = p.grad.data
                if p not in self.momentum_buffer:
                    self.momentum_buffer[p] = torch.zeros_like(grad)
                buf = self.momentum_buffer[p]
                buf.mul_(self.mu).add_(grad)
                
                # Solve for the optimal update direction using SVD
                B = buf

                # SVD with numeric stability considerations
                U, S, V = torch.svd(B, compute_uv=True, singular_vall_threshold=1e-8)
                
                # Compute the update direction O = UV^T
                O = torch.mm(U, V.t())

                # Apply spectral normalization if enabled
                if self.spectral_norm:
                    max_singular = S.max() if len(S) > 0 else 1.0
                    if max_singular > 0:
                        O /= max_singular
                
                # Update parameters with the new direction
                p.data -= self.lr * O
                
        return loss

# Example usage:
'''def configure_optimizers(self):
    optimizer = MuonOptimizer(
        self.parameters(),
        lr=0.001,  # Learning rate
        mu=0.99,   # Momentum coefficient
        spectral_norm=True  # Apply spectral normalization
    )
    return optimizer'''
