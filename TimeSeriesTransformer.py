import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Simulate spots data
np.random.seed(0)
spots = np.random.rand(365, 5)  # 365 days, 5 features (open, close, high, low, volume)

# Split data into train and test sets
train_size = int(len(spots) * 0.8)
spots_train = spots[:train_size]
spots_test = spots[train_size:]

SEQUENCE_SIZE = 10
PREDICTION_SIZE = 3  # Predict the next 3 days

def to_sequences(seq_size, pred_size, obs):
    x = []
    y = []
    for i in range(len(obs) - seq_size - pred_size + 1):
        window = obs[i:(i + seq_size)]
        after_window = obs[(i + seq_size):(i + seq_size + pred_size)]
        x.append(window)
        y.append(after_window)
    return torch.tensor(np.array(x), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

x_train, y_train = to_sequences(SEQUENCE_SIZE, PREDICTION_SIZE, spots_train)
x_test, y_test = to_sequences(SEQUENCE_SIZE, PREDICTION_SIZE, spots_test)

# Create DataLoader
batch_size = 32  # Make it as you want
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, seq_len, pred_len):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, model_dim))
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(model_dim * seq_len, output_dim * pred_len)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = self.transformer_encoder(x)
        x = x.view(x.size(0), -1)  # Flatten for the final fully connected layer
        x = self.fc(x)
        return x.view(x.size(0), pred_len, output_dim)

# Hyperparameters
input_dim = 5  # Number of features in time series data
model_dim = 8  # Dimension of the transformer model (further reduced)
num_heads = 2  # Number of attention heads
num_layers = 1  # Number of transformer layers (reduced)
output_dim = 5  # Number of features to predict (open, close, high, low, volume)
seq_len = SEQUENCE_SIZE  # Length of input sequences
pred_len = PREDICTION_SIZE  # Length of prediction sequences

# Create the model
model = TimeSeriesTransformer(input_dim, model_dim, num_heads, num_layers, output_dim, seq_len, pred_len)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Gradient accumulation steps
accumulation_steps = 4

# Save the best model
best_loss = float('inf')
best_model_path = 'best.pt'

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for i, (x_batch, y_batch) in enumerate(train_loader):
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss = loss / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * accumulation_steps
    
    epoch_loss /= len(train_loader)
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    # Save the best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), best_model_path)

# Testing loop
model.eval()
test_loss = 0
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        test_output = model(x_batch)
        loss = criterion(test_output, y_batch)
        test_loss += loss.item()
test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4f}')
