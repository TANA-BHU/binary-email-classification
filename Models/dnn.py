import torch
import torch.nn as nn
import torch.optim as optim

class DNNModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.layer_2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.layer_3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.dropout1(x)
        x = self.relu(self.layer_2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.layer_3(x))
        return x
