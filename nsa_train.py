import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
from typing import Dict, Any

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-4
EPOCHS = 10
SEQ_LEN = 2880  # Context length from Sundial paper
N_PATCHES = 16  # Patch size from Sundial paper

PDS_SPICE_PATH = "PDS_archive/pds_archive.h5"
ERA5_PATH = "ERA5_archive/era5_archive.h5"
JPL_HORIZONS_PATH = "Horizons_archive/horizons_archive.h5"

class OrbitalDataset(Dataset):
    def __init__(self, data_path: str):
        self.data = np.load(data_path)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        x = torch.FloatTensor(sample['features'])
        y = torch.FloatTensor(sample['labels'])
        
        # Apply re-normalization within each sample
        x_normalized = self.re_normalize(x)
        
        return {
            'x': x_normalized,
            'y': y
        }
    
    def re_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Re-normalize within each sample to mitigate non-stationarity."""
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True)
        return (x - mean) / (std + 1e-8)

class ERA5Dataset(Dataset):
    def __init__(self, data_path: str):
        self.data = np.load(data_path)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        x = torch.FloatTensor(sample['features'])
        y = torch.FloatTensor(sample['labels'])
        
        # Apply re-normalization within each sample
        x_normalized = self.re_normalize(x)
        
        return {
            'x': x_normalized,
            'y': y
        }
    
    def re_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Re-normalize within each sample to mitigate non-stationarity."""
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True)
        return (x - mean) / (std + 1e-8)

class JPLDataset(Dataset):
    def __init__(self, data_path: str):
        self.data = np.load(data_path)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        x = torch.FloatTensor(sample['features'])
        y = torch.FloatTensor(sample['labels'])
        
        # Apply re-normalization within each sample
        x_normalized = self.re_normalize(x)
        
        return {
            'x': x_normalized,
            'y': y
        }
    
    def re_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Re-normalize within each sample to mitigate non-stationarity."""
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True)
        return (x - mean) / (std + 1e-8)

class PatchEmbedding(nn.Module):
    def __init__(self, seq_len: int = SEQ_LEN, patch_size: int = N_PATCHES):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        
        # Embedding layer for patches
        self.embedding = nn.Linear(patch_size, 512)
        
        # Positional embeddings for patches
        self.position_embeddings = nn.Embedding(self.num_patches + 1, 512)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input time series into patch embeddings."""
        batch_size = x.size(0)
        
        # Reshape the input to extract patches
        x_patched = x.view(batch_size, self.num_patches, self.patch_size, -1)
        
        # Compute mean and other features for each patch
        patch_features = torch.mean(x_patched, dim=2)  # Simplified example
        
        # Embed the patch features
        embedded = self.embedding(patch_features)
        
        # Add positional embeddings
        positions = torch.arange(self.num_patches).expand(batch_size, -1).to(x.device)
        pos_embeddings = self.position_embeddings(positions)
        embedded += pos_embeddings.unsqueeze(0)
        
        return embedded.permute(0, 2, 1)  # (batch_size, embed_dim, num_patches)

class NSparseAttention(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Parameter vector 𝜑 for sparse attention
        self.phi = nn.Parameter(torch.randn(1, self.num_heads, self.head_dim))
        
        # Positional embeddings for temporal modeling
        self.position_embeddings = nn.Embedding(SEQ_LEN, embed_dim)

    def apply_sparse_attention(self, attn_probs: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply dynamic hierarchical sparse attention pattern."""
        # Coarse-grained token compression
        block_size = 64  # Adjust based on your requirements
        num_blocks = (seq_len + block_size - 1) // block_size
        
        # Initialize coarse mask for intra-block attention
        coarse_mask = torch.zeros(seq_len, seq_len).bool()
        for i in range(num_blocks):
            start = i * block_size
            end = min((i + 1) * block_size, seq_len)
            coarse_mask[start:end, start:end] = True
        
        # Fine-grained token selection using phi
        fine_mask = torch.zeros_like(coarse_mask)
        for i in range(num_blocks):
            start = i * block_size
            end = min((i + 1) * block_size, seq_len)
            
            # Select specific tokens within the block based on phi
            q_block = self.phi[:, :, :end - start]  # Adjust indices as needed
            similarity_scores = torch.bmm(q_block.unsqueeze(2), q_block.unsqueeze(1))
            top_k = min(8, end - start)  # Select top-k similar tokens
            
            # Get indices of top-k scores
            _, indices = torch.topk(similarity_scores.squeeze(), k=top_k, dim=0)
            
            # Mark selected positions in the fine mask
            for idx in indices:
                if idx < (end - start):
                    fine_mask[start + idx] = True
        
        # Combine coarse and fine masks
        combined_mask = coarse_mask & fine_mask
        
        # Apply the combined mask to attention probabilities
        attn_probs[~combined_mask] = -torch.inf  # Zero out non-selected positions
        
        return attn_probs

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with native sparse attention."""
        batch_size, seq_len, _ = x.size()
        
        # Incorporate positional embeddings
        pos_embeddings = self.position_embeddings(torch.arange(seq_len, device=x.device))
        x_with_pos = x + pos_embeddings.unsqueeze(0)
        
        # Compute query vectors using phi
        q = x_with_pos + self.phi  # (batch_size, seq_len, head_dim)
        
        # Reshape for block-wise processing
        '''q_reshaped = q.view(batch_size, num_blocks, block_size, self.head_dim)
        
        # Compute attention scores in blocks
        attn_scores = torch.bmm(q_reshaped.unsqueeze(2), q_reshaped.unsqueeze(1)) / np.sqrt(self.head_dim)
        
        # Apply softmax to get probability distribution (block-wise)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Apply sparse attention pattern
        attn_probs = self.apply_sparse_attention(attn_probs.view(batch_size, seq_len, seq_len), seq_len)'''

        # Compute attention scores
        attn_scores = torch.bmm(q.unsqueeze(2), q.unsqueeze(1)) / np.sqrt(self.head_dim)
        
        # Apply softmax to get probability distribution
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Apply sparse attention pattern (dynamic hierarchical sparsity)
        attn_probs = self.apply_sparse_attention(attn_probs, seq_len)
        
        # Compute the final output using the attention mechanism
        out = torch.bmm(x.unsqueeze(2), attn_probs.permute(0, 2, 1))[:, :, 0]
        
        return out


class TimeFlowLoss(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.flow_params = nn.Parameter(torch.randn(d_model))
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass with TimeFlow Loss."""
        batch_size, seq_len, _ = x.size()
        
        # Compute flow probabilities
        flow_logits = x + self.flow_params.unsqueeze(1).unsqueeze(2)
        flow_probs = torch.sigmoid(flow_logits)
        
        # Calculate the loss based on flow matching
        loss = torch.mean((flow_probs - y.unsqueeze(-1)) ** 2)
        
        return loss

def create_stratified_dataloaders(datasets: Dict[str, Dataset]) -> Dict[str, DataLoader]:
    """Create balanced dataloaders using stratified sampling."""
    dataloaders = {}
    
    # Stratified k-fold for each dataset
    for name, dataset in datasets.items():
        train_indices = list(range(len(dataset)))
        np.random.shuffle(train_indices)
        
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices[:-100])
        val_sampler = torch.utils.data.SubsetRandomSampler(train_indices[-100:])
        
        dataloaders[name] = DataLoader(
            dataset,
            batch_size=64,
            sampler=train_sampler if name != 'test' else val_sampler,
            num_workers=4
        )
    
    return dataloaders

def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    dataloaders: Dict[str, DataLoader],
    device: str,
    num_epochs: int = EPOCHS,
) -> None:
    """Train the model with balanced sampling and evaluation."""
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        for batch_idx, (x, y) in enumerate(dataloaders['train']):
            x = x.to(device)
            y = y.to(device)
            
            # Apply patch embeddings and re-normalization
            x_patched = PatchEmbedding()(x.permute(0, 2, 1))  # (batch_size, embed_dim, num_patches)
            x_embedded = x_patched.permute(0, 2, 1)  # (batch_size, num_patches, embed_dim)
            
            # Forward pass with sparse attention
            outputs = model(x_embedded)  # (batch_size, num_patches, embed_dim)
            
            # Reshape for loss calculation
            outputs_reshaped = outputs.permute(0, 2, 1)  # (batch_size, embed_dim, num_patches)
            
            loss = criterion(outputs_reshaped, y.permute(0, 2, 1))
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in dataloaders['test']:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                
                # Apply patch embeddings and re-normalization
                x_patched_val = PatchEmbedding()(x_val.permute(0, 2, 1))  # (batch_size, embed_dim, num_patches)
                x_embedded_val = x_patched_val.permute(0, 2, 1)  # (batch_size, num_patches, embed_dim)
                
                outputs_val = model(x_embedded_val)  # (batch_size, num_patches, embed_dim)
                
                # Reshape for loss calculation
                outputs_reshaped_val = outputs_val.permute(0, 2, 1)  # (batch_size, embed_dim, num_patches)
                
                loss_val = criterion(outputs_reshaped_val, y_val.permute(0, 2, 1))
                
                val_loss += loss_val.item()
        
        avg_val_loss = val_loss / len(dataloaders['test'])
        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {avg_val_loss}")
        
        # Save model if best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Calculate additional metrics
    metric_mae = nn.L1Loss()
    metric_rmse = lambda x, y: torch.sqrt(torch.mean((x - y) ** 2))
    metric_crps = lambda x, y: torch.mean(torch.abs(x - y))
    metric_wql = lambda x, y: torch.mean((x - y) ** 2)
    
    with torch.no_grad():
        total_mae = 0.0
        total_rmse = 0.0
        total_crps = 0.0
        total_wql = 0.0
        
        for x_val, y_val in dataloaders['test']:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            
            # Generate multiple predictions (e.g., 20 samples)
            samples = [model(x_val.permute(0, 2, 1)) for _ in range(20)]
            preds = torch.stack(samples, dim=0)  # (num_samples, batch_size, seq_len, embed_dim)
            
            mae = metric_mae(preds.mean(dim=0), y_val.permute(0, 2, 1))
            rmse = metric_rmse(preds.mean(dim=0), y_val.permute(0, 2, 1))
            crps = metric_crps(preds, y_val.permute(0, 2, 1).unsqueeze(0).expand_as(preds))
            wql = metric_wql(preds, y_val.permute(0, 2, 1).unsqueeze(0).expand_as(preds))
            
            total_mae += mae.item()
            total_rmse += rmse.item()
            total_crps += crps.item()
            total_wql += wql.item()
    
    avg_mae = total_mae / len(dataloaders['test'])
    avg_rmse = total_rmse / len(dataloaders['test'])
    avg_crps = total_crps / len(dataloaders['test'])
    avg_wql = total_wql / len(dataloaders['test'])
    
    print(f"Final Metrics - MAE: {avg_mae:.4f}, RMSE: {avg_rmse:.4f}, CRPS: {avg_crps:.4f}, WQL: {avg_wql:.4f}")

# Initialize datasets
pds_dataset = OrbitalDataset(PDS_SPICE_PATH)
era5_dataset = ERA5Dataset(ERA5_PATH)
jpl_dataset = JPLDataset(JPL_HORIZONS_PATH)

# Combine datasets for stratified sampling
datasets = {
    'PDS': pds_dataset,
    'ERA5': era5_dataset,
    'JPL': jpl_dataset,
}

# Create balanced dataloaders
dataloaders = create_stratified_dataloaders(datasets)

# Initialize model and optimizer
model = NSparseAttention(embed_dim=512, num_heads=8).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = TimeFlowLoss(d_model=512)

# Train the model
train_model(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    dataloaders=dataloaders,
    device=DEVICE,
    num_epochs=EPOCHS
)
