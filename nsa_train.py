import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from sklearn.preprocessing import RobustScaler
from scipy.interpolate import interp1d

# Constants
HDF5_FILE_SUFFIX = ".h5"
DEFAULT_HDF5_DIR = "preprocessed_data"
BATCH_SIZE = 64
NUM_WORKERS = 4
PIN_MEMORY = True
SHUFFLE = True
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class OrbitalDataset(Dataset):
    """
    Custom dataset class for orbital time series data.
    """

    def __init__(self, data: np.ndarray, seq_length: int = 32):
        self.data = data
        self.seq_length = seq_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Retrieve a single sample with sequence length.
        """
        start_idx = idx
        end_idx = idx + self.seq_length
        
        if end_idx > len(self.data):
            end_idx = len(self.data)
        
        # Get positions and velocities
        positions = self.data[idx, :3]
        velocities = self.data[idx, 3:]
        
        # Interpolate missing values
        mask_pos = ~np.isnan(positions)
        mask_vel = ~np.isnan(velocities)
        
        if np.sum(mask_pos) == 0:
            positions = np.zeros_like(positions)
        else:
            positions = interp1d(mask_pos, positions[mask_pos])(idx)
        
        if np.sum(mask_vel) == 0:
            velocities = np.zeros_like(velocities)
        else:
            velocities = interp1d(mask_vel, velocities[mask_vel])(idx)
        
        return {
            "positions": torch.tensor(positions, dtype=torch.float32),
            "velocities": torch.tensor(velocities, dtype=torch.float32)
        }

class TimeSeriesDataModule:
    """
    Data module for handling orbital time series data.
    """

    def __init__(self, train_data_paths: List[str], val_data_paths: Optional[List[str]] = None):
        self.train_data_paths = train_data_paths
        self.val_data_paths = val_data_paths

    def _load_hdf5(self, path: str) -> np.ndarray:
        """
        Load data from HDF5 file.
        
        Args:
            path: Path to the HDF5 file
            
        Returns:
            numpy array containing the time series data
        """
        return np.load(path)

    def setup(self) -> None:
        """
        Set up datasets with balanced sampling.
        """
        # Load training data
        if not self.train_data_paths:
            raise FileNotFoundError("Training data paths are empty.")
        
        train_datasets = []
        for path in self.train_data_paths:
            dataset = OrbitalDataset(
                data=self._load_hdf5(path),
                seq_length=32  # Adjust sequence length as needed
            )
            train_datasets.append(dataset)
        
        # Split into training, validation, and testing sets with balanced sampling
        total_samples = sum(len(d) for d in train_datasets)
        train_size = int(0.6 * total_samples)
        val_size = int(0.1 * total_samples)
        test_size = total_samples - train_size - val_size
        
        # Stratified split to preserve source distribution
        train_dataset, val_dataset, test_dataset = random_split(
            ConcatDataset(train_datasets),
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(SEED)
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            shuffle=SHUFFLE,
            drop_last=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            shuffle=False,
            drop_last=True
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            shuffle=False,
            drop_last=True
        )

    def train_dataloader(self) -> DataLoader:
        """
        Return the training dataloader.
        """
        return self.train_loader

    def val_dataloader(self) -> Optional[DataLoader]:
        """
        Return the validation dataloader.
        """
        return self.val_loader

    def test_dataloader(self) -> DataLoader:
        """
        Return the testing dataloader.
        """
        return self.test_loader

class NativeSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size=64, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.dropout = dropout
        
        # Key and value projections
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout layers
        self.dropout_layer = nn.Dropout(dropout)
    
    def _compute_attention(self, x):
        """
        Compute attention scores using native sparse attention mechanism.
        
        Args:
            x: Input tensor [seq_len, batch_size, embed_dim]
            
        Returns:
            Attention output tensor [seq_len, batch_size, embed_dim]
        """
        # Project keys and values
        k = self.key_proj(x)
        v = self.value_proj(x)
        
        # Compute attention scores
        seq_len = x.size(0)
        attention_scores = torch.bmm(k.transpose(-2, -1), v)
        
        # Apply softmax and dropout
        attention_probs = nn.Softmax(dim=-1)(attention_scores / (seq_len ** 0.5))
        attention_probs = self.dropout_layer(attention_probs)
        
        # Compute output
        output = torch.bmm(attention_probs, x.transpose(-2, -1)).transpose(-2, -1)
        
        return output
    
    def forward(self, x):
        """
        Forward pass of the NativeSparseAttention module.
        
        Args:
            x: Input tensor [seq_len, batch_size, embed_dim]
            
        Returns:
            Output tensor [seq_len, batch_size, embed_dim]
        """
        return self._compute_attention(x)

class Transformer(nn.Module):
    def __init__(self, input_size=6, hidden_size=512, nhead=4):
        super().__init__()
        
        # Input embedding
        self.embedding = nn.Linear(input_size, hidden_size)
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 32, hidden_size))
        
        # Transformer layers
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=nhead,
            num_encoder_layers=6,
            num_decoder_layers=6
        )
        
        # Flow-matching components
        self.flow_matching_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        
        # Heads for prediction (mean and variance)
        self.mean_head = nn.Linear(hidden_size, input_size)
        self.variance_head = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        """
        Forward pass of the transformer model.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            
        Returns:
            Dict containing mean and variance predictions
        """
        batch_size, seq_len, input_size = x.size()
        
        # Embedding
        x = self.embedding(x.view(-1, input_size)).view(batch_size, seq_len, -1)
        
        # Add positional encoding
        pos_emb = self.pos_embedding.expand(batch_size, -1, -1)
        x = torch.cat((x, pos_emb), dim=-1)
        
        # Transformer
        x = self.transformer(x.permute(1, 0, 2))
        x = x.permute(1, 0, 2)
        
        # Flow-matching
        flow_features = self.flow_matching_net(x)
        
        # Predict mean and variance
        mean = self.mean_head(flow_features)
        var = torch.nn.functional.softplus(self.variance_head(flow_features)) + 1e-6
        
        return {
            "mean": mean,
            "var": var
        }

class TrainingLoop:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        
    def train_step(self, batch):
        """
        Single training step.
        
        Args:
            batch: Dict containing "positions" and "velocities"
            
        Returns:
            Loss value
        """
        x = batch["positions"].float()
        y = batch["velocities"].float()
        
        # Forward pass
        outputs = self.model(x)
        
        # Calculate NLL loss
        mean_pred = outputs["mean"]
        var_pred = outputs["var"]
        
        nll_loss = torch.mean(-0.5 * torch.log(var_pred) - 0.5 * (y - mean_pred).pow(2) / var_pred)
        
        # Add flow-matching regularization
        flow_features = self.model.flow_matching_net(x)
        flow_loss = torch.nn.functional.mse_loss(flow_features, x)
        
        total_loss = nll_loss + flow_loss
        
        return total_loss

    def train_epoch(self, dataloader):
        """
        Single training epoch.
        
        Args:
            dataloader: DataLoader for the training data
        """
        self.model.train()
        for batch in dataloader:
            loss = self.train_step(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

class Evaluation:
    def __init__(self, model):
        self.model = model
        
    def evaluate(self, dataloader, metrics=['MAE', 'RMSE', 'CRPS']):
        """
        Evaluate the model using specified metrics.
        
        Args:
            dataloader: DataLoader for evaluation data
            metrics: List of metrics to compute
            
        Returns:
            Dictionary containing metric values
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        
        for batch in dataloader:
            with torch.no_grad():
                outputs = self.model(batch["positions"].float())
                
            mean_pred = outputs["mean"].cpu().numpy()
            var_pred = outputs["var"].cpu().numpy()
            y_true = batch["velocities"].cpu().numpy()
            
            all_preds.append(mean_pred)
            all_targets.append(y_true)
        
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        
        results = {}
        if 'MAE' in metrics:
            mae = np.mean(np.abs(y_pred - y_true))
            results['MAE'] = mae
        if 'RMSE' in metrics:
            rmse = np.sqrt(np.mean((y_pred - y_true)**2))
            results['RMSE'] = rmse
        if 'CRPS' in metrics:
            crps = np.mean( (np.sign(y_pred - y_true) *
                            np.sqrt(np.abs(y_pred - y_true))) )
            results['CRPS'] = crps
        
        return results

def main():
    """
    Main function to train the model with multi-source balanced sampling.
    """
    # Initialize logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create data module with all training and validation paths
    train_data_paths = [
        os.path.join(DEFAULT_HDF5_DIR, "pds_spice.h5"),
        os.path.join(DEFAULT_HDF5_DIR, "era5.h5"),
        os.path.join(DEFAULT_HDF5_DIR, "jpl_horizons.h5")
    ]
    
    data_module = TimeSeriesDataModule(train_data_paths=train_data_paths)
    data_module.setup()
    
    # Create model
    input_size = 6  # positions (3) + velocities (3)
    model = Transformer(input_size=input_size)
    
    # Set up optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # Initialize training loop
    trainer = TrainingLoop(model=model, criterion=criterion, optimizer=optimizer)
    
    # Train model
    for epoch in range(100):
        logging.info(f"Epoch {epoch + 1}/100")
        trainer.train_epoch(data_module.train_loader)
    
    # Evaluation
    val_results = Evaluation(model).evaluate(data_module.val_loader)
    logging.info(f"Validation Results: {val_results}")
    
    # Save trained model (optional)
    torch.save(model.state_dict(), 'edi_model.pth')

if __name__ == "__main__":
    main()
