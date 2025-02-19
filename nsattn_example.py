import torch
import math
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


# Still need to frankenstein this into nsa_train.py for included flow matching, 
# timeflowloss, and split of train, test, validation sets

# also this could still be wrong idk winging it

# Define Compressor class (hardware-aligned token compression)
class Compressor(torch.nn.Module):
    def __init__(self, d_h: int):
        super().__init__()
        self.d_h = d_h
        # Learnable parameters for non-linear transformation
        self.W_c = torch.nn.Parameter(torch.randn(d_h, d_h // 2))
        self.b_c = torch.nn.Parameter(torch.zeros(d_h // 2))
        # Layer normalization for stability
        self.ln = torch.nn.LayerNorm(d_h // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_h]
        Returns:
            Compressed tensor [batch_size, seq_len, d_h//2]
        """
        # Apply non-linear transformation
        x_compressed = (x @ self.W_c + self.b_c)
        x_compressed = torch.relu(x_compressed)
        x_compressed = self.ln(x_compressed)
        return x_compressed

# Define Gate class (hardware-aligned gate mechanism)
class Gate(torch.nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.dim = dim
        self.heads = heads
        # Learnable parameters for attention control
        self.W_g = torch.nn.Parameter(torch.randn(dim, 1))
        self.scaling = torch.nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, dim]
        Returns:
            Gate output [batch_size, seq_len, heads]
        """
        # Compute global context-aware gate
        g = (x @ self.W_g).squeeze(-1)
        g = torch.sigmoid(g * self.scaling)
        g = g.unsqueeze(-1)  # [batch_size, seq_len, 1]
        return g

# Define Positional Embedding class (hardware-aligned PE)
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.dropout = 0.1
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        Returns:
            Positional embeddings added to input
        """
        x = x + self.pe[:x.size(1)]
        return F.dropout(x, p=self.dropout, training=self.training)

# Define Native Sparse Attention (NSA) class
class NSA(torch.nn.Module):
    def __init__(self, dim: int, heads: int, d_h: int = 64):
        super().__init__()
        self.h = heads
        self.d_h = d_h
        # Query/key/value transformations
        self.wq = torch.nn.Linear(dim, heads * d_h)
        self.wk = torch.nn.Linear(dim, heads * d_h)
        self.wv = torch.nn.Linear(dim, heads * d_h)
        self.wo = torch.nn.Linear(heads * d_h, dim)
        # Gate mechanism
        self.gate = Gate(dim, heads)
        # Token compression modules
        self.f_cmp_k = Compressor(d_h)
        self.f_cmp_v = Compressor(d_h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, dim]
        Returns:
            Output tensor after NSA computation
        """
        # Compute query, key, value
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        # Reshape for multi-head attention
        q, k, v = [t.unflatten(-1, (self.h, -1)).transpose(-2, -3) for t in [q, k, v]]
        
        # Compute gate
        g = self.gate(x)
        
        # Compress keys and values
        k_cmp = self.f_cmp_k(k)
        v_cmp = self.f_cmp_v(v)
        
        # Compute attention scores
        attn_score = (q @ k_cmp.transpose(-2, -1)) * g
        
        # Apply softmax with hardware-aligned efficiency
        attn_score = F.softmax(attn_score, dim=-1)
        
        # Compute output
        out = attn_score @ v_cmp
        
        # Reshape back and apply output weights
        out = out.transpose(-2, -3).flatten(-2)
        return self.wo(out)

# Define Transformer Block with NSA
class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.positional_embedding = PositionalEmbedding(d_model)
        self.self_attention = NSA(d_model=d_model, heads=num_heads)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        Returns:
            Output tensor after transformer block
        """
        x = self.positional_embedding(x)
        x = self.self_attention(x)
        x = self.dropout(x)
        return x

# Define Orbital Dataset class
class OrbitalDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, seq_length: int = 32):
        # Load orbital data and preprocess
        self.data = pd.read_csv(data_path)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length + 1

    def __getitem__(self, idx):
        # Return input features and target (e.g., future positions/velocities)
        inputs = self.data.iloc[idx:idx + self.seq_length]
        target_idx = idx + self.seq_length
        target = self.data.iloc[target_idx] if target_idx < len(self.data) else None
        return inputs, target

# Training script
def main():
    # Initialize model, optimizer, and data loader
    d_model = 512
    num_heads = 8
    dropout = 0.1
    batch_size = 32
    learning_rate = 0.001
    seq_length = 32

    model = TransformerBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    # Load orbital data
    dataset = OrbitalDataset(data_path='path_to_orbital_data.csv', seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            inputs, targets = batch
            outputs = model(inputs)
            
            # Compute loss (e.g., using MSE for regression)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

if __name__ == "__main__":
    main()
