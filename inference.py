import os
import logging
import numpy as np
import torch
from pytorch_lightning import LightningModule
from typing import Dict, List, Optional
from sklearn.preprocessing import RobustScaler
from scipy.stats import norm

# Add necessary imports for visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Ensure compatibility with your trained model architecture
class TransformerTimeSeriesModel(LightningModule):
    def __init__(self, input_size: int = 6, patch_size: int = 16, context_length: int = 2880, prediction_length: int = 720):
        super().__init__()
        
        # Tokenization and embedding
        self.patch_embedding = torch.nn.Linear(input_size, patch_size)
        self.positional_encoding = torch.nn.Embedding(context_length, patch_size)
        
        # Transformer layers
        self.transformer_layers = torch.nn.ModuleList([
            torch.nn.TransformerDecoderLayer(
                d_model=patch_size,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                activation='gelu'
            )
            for _ in range(6)
        ])
        
        # Final prediction head
        self.prediction_head = torch.nn.Linear(patch_size, input_size)
        
    def forward(self, x: torch.Tensor) -> Dict:
        """
        Forward pass for the TransformerTimeSeriesModel.
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_size]
        Returns:
            Dict containing 'mean' and 'variance' tensors
        """
        # Tokenization
        x = self.patch_embedding(x)
        
        # Positional encoding
        positions = torch.arange(0, x.size(1), dtype=torch.long).unsqueeze(0).repeat(x.size(0), 1)
        x += self.positional_encoding(positions)
        
        # Transformer decoding
        for layer in self.transformer_layers:
            x = layer(x)
            
        # Generate predictions
        outputs = self.prediction_head(x)
        
        # Calculate variance (example implementation)
        mean = torch.mean(outputs, dim=-1, keepdim=True)
        var = torch.var(torch.cat([outputs - mean, mean], dim=-1), dim=-1, keepdim=True)
        
        return {
            "mean": outputs,
            "variance": var
        }

# Initialize the Flask app
app = Flask(__name__)

def load_and_predict(model_path: str, input_data: np.ndarray) -> Dict:
    """
    Load the trained model and generate predictions for live time series data.
    Incorporates uncertainty metrics and returns formatted results.
    """
    try:
        # Load the trained model
        model = TransformerTimeSeriesModel(
            input_size=6,
            patch_size=16,
            context_length=2880,
            prediction_length=720
        )
        
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Preprocess input data (if needed, e.g., normalization)
        scaler = RobustScaler()
        normalized_data = scaler.fit_transform(input_data)
        
        # Reshape input for the Transformer model
        batch_size = 1
        seq_length = input_data.shape[0]
        input_tensor = torch.tensor(normalized_data.reshape(1, seq_length, -1), dtype=torch.float32)
        
        # Generate predictions using the trained model
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # Extract mean and variance (if your model outputs distributions)
        mean_predictions = outputs["mean"].numpy().squeeze()
        var_predictions = outputs["variance"].numpy().squeeze()

        # Calculate uncertainty metrics
        std_predictions = np.sqrt(var_predictions)
        
        # Propagate uncertainties (example: using standard deviation for quantiles)
        lower_quantile = mean_predictions - 1.96 * std_predictions
        upper_quantile = mean_predictions + 1.96 * std_predictions
        
        return {
            "mean": mean_predictions.tolist(),
            "variance": var_predictions.tolist(),
            "std_deviation": std_predictions.tolist(),
            "lower_95ci": lower_quantile.tolist(),
            "upper_95ci": upper_quantile.tolist()
        }
        
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return {"error": str(e)}

def create_inference_sequences(data: np.ndarray, seq_length: int) -> np.ndarray:
    """
    Create sequences from input data for model inference.
    """
    num_samples = len(data)
    sequences = []
    
    for i in range(num_samples - seq_length + 1):
        sequence = data[i:i+seq_length]
        sequences.append(sequence)
    
    return np.array(sequences)

def visualize_predictions(true_positions: List[float], 
                        predicted_positions: List[float],
                        uncertainties: List[float]) -> None:
    """
    Visualize true and predicted positions with uncertainty intervals.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(true_positions)), true_positions, label='True Positions', linestyle='--')
    plt.plot(range(len(predicted_positions)), predicted_positions, label='Predicted Positions')
    plt.fill_between(
        range(len(predicted_positions)),
        predicted_positions - uncertainties,
        predicted_positions + uncertainties,
        color='gray',
        alpha=0.3,
        label='Uncertainty Interval'
    )
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.title('True vs Predicted Positions with Uncertainty')
    plt.legend()
    plt.show()

def main_inference():
    """
    Main function for generating predictions and visualization.
    """
    try:
        # Example usage
        model_path = 'path_to_your_trained_model.ckpt'
        input_data = np.array([...])  # Shape (seq_length, input_size)
        
        # Generate inference sequences
        sequences = create_inference_sequences(input_data, seq_length=2880)
        
        # Process each sequence
        all_predictions = []
        for seq in sequences:
            predictions = load_and_predict(model_path, seq)
            if 'error' in predictions:
                logging.error(f"Error processing sequence: {predictions['error']}")
                continue
            all_predictions.append(predictions)
        
        # Extract results for visualization
        true_positions = [...]  # Populate with actual values
        predicted_positions = [p['mean'] for p in all_predictions]
        uncertainties = [p['std_deviation'] for p in all_predictions]
        
        visualize_predictions(true_positions, predicted_positions, uncertainties)
        
    except Exception as e:
        logging.error(f"Failed to run inference: {str(e)}")

if __name__ == "__main__":
    main_inference()
