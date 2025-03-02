import os
import logging
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from sklearn.preprocessing import RobustScaler
from scipy.interpolate import interp1d
from typing import Dict, List, Optional
from modules.MuonOptimizer import MuonOptimizer
# Constants
DEFAULT_DATA_DIR = "preprocess"
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

class TimeSeriesDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for multi-source time series forecasting.
    """

    def __init__(self, train_data_paths: List[str], val_data_paths: Optional[List[str]] = None):
        super().__init__()
        self.train_data_paths = train_data_paths
        self.val_data_paths = val_data_paths

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
                data=np.load(path),
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

class TransformerTimeSeriesModel(pl.LightningModule):
    """
    Transformer-based time series forecasting model inspired by Sundial with flow-matching components.
    """

    def __init__(self, input_size: int = 6, hidden_size: int = 512, nhead: int = 4):
        super().__init__()
        
        # Input size includes positions (3) and velocities (3)
        self.embedding = torch.nn.Linear(input_size, hidden_size)
        
        # Transformer layers
        self.transformer = torch.nn.Transformer(
            d_model=hidden_size,
            nhead=nhead,
            num_encoder_layers=6,
            num_decoder_layers=6
        )
        
        # Flow-matching components
        self.flow_matching_net = nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_size)
        )
        
        # Heads for prediction (mean and variance)
        self.mean_head = torch.nn.Linear(hidden_size, input_size)
        self.variance_head = torch.nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> Dict:
        """
        Forward pass of the model.
        """
        batch_size, seq_len, input_size = x.size()
        
        # Embedding
        x = self.embedding(x.view(-1, input_size)).view(batch_size, seq_len, -1)
        
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

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        """
        Training step with TimeFlow Loss.
        """
        # Get data from batch
        if not isinstance(batch, Dict):
            raise ValueError("Batch must be a dictionary.")
        
        x = batch["positions"].float()
        y = batch["velocities"].float()
        
        # Forward pass
        outputs = self.forward(x)
        
        # Calculate TimeFlow Loss
        mean_pred = outputs["mean"]
        var_pred = outputs["var"]
        
        nll_loss = torch.mean(-0.5 * torch.log(var_pred) - 0.5 * (y - mean_pred).pow(2) / var_pred)
        
        # Add flow-matching regularization
        flow_features = self.flow_matching_net(x)
        flow_loss = torch.nn.functional.mse_loss(flow_features, x)
        
        total_loss = nll_loss + flow_loss
        
        self.log("train_loss", total_loss, prog_bar=True)
        
        return {"loss": total_loss}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer.
        """
        optimizer = MuonOptimizer(self.parameters(), lr=0.001, mu=0.99)
        return optimizer
    

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
        os.path.join(DEFAULT_DATA_DIR, "pds_spice.parquet"),
        os.path.join(DEFAULT_DATA_DIR, "era5.parquet"),
        os.path.join(DEFAULT_DATA_DIR, "jpl_horizons.parquet")
    ]
    
    data_module = TimeSeriesDataModule(train_data_paths=train_data_paths)
    data_module.setup()
    
    # Create model
    input_size = 6  # positions (3) + velocities (3)
    model = TransformerTimeSeriesModel(input_size=input_size)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=100,
        gpus=torch.cuda.device_count(),
        num_workers=NUM_WORKERS,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=10),
            pl.callbacks.ModelCheckpoint(save_top_k=3, monitor="val_loss")
        ]
    )
    
    # Train model
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()
