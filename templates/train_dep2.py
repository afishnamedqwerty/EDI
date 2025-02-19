import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import crps

# 1. Data Handling and Preprocessing
class OrbitalDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_length=32, target_size=1):
        self.data = data
        self.seq_length = seq_length
        self.target_size = target_size
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length:idx + self.seq_length + self.target_size]
        
        # Normalize using RobustScaler
        scaler = RobustScaler()
        normalized_x = torch.FloatTensor(scaler.fit_transform(x))
        normalized_y = torch.FloatTensor(scaler.transform(y))
        
        return normalized_x, normalized_y

# 2. Model Architecture with Flow-Matching
class OrbitalPredictionModel(torch.nn.Module):
    def __init__(self, input_size=6, hidden_size=512, nhead=4):
        super(OrbitalPredictionModel, self).__init__()
        self.embedding = torch.nn.Linear(input_size, hidden_size)
        self.transformer_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.embedding(x.view(-1, x.size(-1)))
        x = x.view(batch_size, -1, x.size(-1))
        x = x.permute(1, 0, 2)  # [seq_len, batch, d_model]
        x = self.transformer_layer(x)
        x = x.permute(1, 0, 2)  # [batch, seq_len, d_model]
        
        return x

# 3. TimeFlow Loss Implementation
class TimeFlowLoss(torch.nn.Module):
    def __init__(self, n_dims=6, hidden_size=512):
        super(TimeFlowLoss, self).__init__()
        self.flow_network = torch.nn.Sequential(
            torch.nn.Linear(n_dims + hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, n_dims)
        )
        
    def forward(self, inputs, outputs):
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        
        # Embed the input
        x = self.embedding(inputs.view(-1, inputs.size(-1)))
        x = x.view(batch_size, seq_len, -1)
        
        # Compute flow features
        flow_features = torch.cat((x, outputs), dim=-1)
        flows = self.flow_network(flow_features.view(-1, flow_features.size(-1)))
        
        # Calculate the loss based on flow matching
        loss = torch.mean(torch.abs(flows[:, :3] - flows[:, 3:]))
        
        return loss

# 4. Training Setup with Flow-Matching
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=100):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_features, labels in train_loader:
            outputs = model(batch_features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_features, labels in val_loader:
                outputs = model(batch_features)
                val_loss += criterion(outputs, labels).item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}')

# 5. Evaluation Metrics
def evaluate_model(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch_features, labels in loader:
            outputs = model(batch_features)
            
            y_true.append(labels.cpu().numpy())
            y_pred.append(outputs.cpu().numpy())
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)**0.5
    crps_score = crps.crps(y_true, y_pred)
    
    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'CRPS: {crps_score:.4f}')

# 6. Main Training Script
if __name__ == '__main__':
    # Parameters
    seq_length = 32
    batch_size = 64
    num_epochs = 100
    
    # Load h5 datasets (assuming they are preprocessed)
    train_dataset = np.load('train_data.h5')
    val_dataset = np.load('val_data.h5')
    
    # Create dataset and dataloaders
    train_loader = DataLoader(OrbitalDataset(train_dataset, seq_length=seq_length), 
                             batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(OrbitalDataset(val_dataset, seq_length=seq_length), 
                            batch_size=batch_size, shuffle=False)
    
    # Initialize model and optimizer
    input_size = 6  # positions (3) + velocities (3)
    hidden_size = 512
    nhead = 4
    
    model = OrbitalPredictionModel(input_size=input_size, hidden_size=hidden_size, nhead=nhead)
    criterion = TimeFlowLoss(n_dims=6, hidden_size=hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Train model
    train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=num_epochs)
    
    # Evaluate model
    evaluate_model(model, val_loader)

