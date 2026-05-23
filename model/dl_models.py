import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    """
    1-Dimensional Convolutional Network engineered to extract independent
    spatial variations from overlapping sliding windows.
    """
    def __init__(self, in_channels=7, num_classes=3):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x expected shape: (Batch, Channels, Time) => (B, 7, 640)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.pool(x).squeeze(-1) # shape: (B, 256)
        x = self.dropout(x)
        return self.fc(x)

class LSTMNet(nn.Module):
    """
    Bidirectional LSTM highly optimized for discovering extreme 
    temporal constraints deep inside the unified WESAD physiological matrices.
    """
    def __init__(self, in_channels=7, hidden_size=64, num_layers=2, num_classes=3):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.dropout = nn.Dropout(0.4)
        # Bidirectional means hidden layer is 2 * hidden_size
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # x is (B, C, T), LSTM expects (B, T, C) if batch_first=True
        x_mapped = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x_mapped)
        
        # Pull the last time step output to push to linear layout
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        return self.fc(last_out)

class HybridCNN_LSTM(nn.Module):
    """
    Advanced Multi-Modal Hybrid Architecture.
    Yields maximum performance by crunching the feature vectors through CNNs
    first, and dumping the abstracted feature-states into the B-LSTM.
    """
    def __init__(self, in_channels=7, num_classes=3):
        super(HybridCNN_LSTM, self).__init__()
        
        # Spatial Feature Extractor
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Temporal Mapping Engine
        self.lstm = nn.LSTM(
            input_size=128, # Match CNN output channel
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Convolution Extraction -> (B, 7, 640)
        c = F.relu(self.bn1(self.conv1(x)))
        c = self.pool1(c)
        c = F.relu(self.bn2(self.conv2(c)))
        c = self.pool2(c) 
        # c shape is now ~ (B, 128, 80)
        
        # Prepare for LSTM
        c = c.transpose(1, 2) # (B, Time, Channels)
        
        lstm_out, _ = self.lstm(c)
        last_out = lstm_out[:, -1, :] # Pull last hidden payload
        
        d = self.dropout(last_out)
        f = F.relu(self.fc1(d))
        return self.fc2(f)
