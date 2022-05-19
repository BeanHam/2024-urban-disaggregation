import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSR(nn.Module):
    
    """
    Super-Resolution of graphs
    
    """
    
    def __init__(self, input_dim, output_dim):
        
        super().__init__()
        
        self.lstm = nn.LSTM(input_dim, output_dim, batch_first=True)
        
    def forward(self, att):
        
        ## down sampling layers
        output, (hn, cn) = self.lstm(att)
        
        return output, hn, cn