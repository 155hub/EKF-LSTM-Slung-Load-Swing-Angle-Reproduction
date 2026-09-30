import torch
import torch.nn as nn

class LSTMCorrectionModel(nn.Module):
    def __init__(self, input_dim=13, state_dim=7, hidden_dim=64, num_layers=2):
        super(LSTMCorrectionModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, state_dim)
        
        # Initialize output to be small
        with torch.no_grad():
            self.fc.weight.mul_(1e-3)
            self.fc.bias.zero_()

    def forward(self, x, hidden=None):
        """
        x: (Batch, Seq_Len, Input_Dim)
        hidden: (h_0, c_0) tuple or None
        """
        out, hidden = self.lstm(x, hidden)
        
        # out is (Batch, Seq_Len, Hidden_Dim)
        correction = self.fc(out) 
        
        return correction, hidden
