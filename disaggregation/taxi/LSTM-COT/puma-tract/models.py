import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

## seed
seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True

# ---------------------
# weights initialization
# ---------------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.01)
        torch.nn.init.constant_(m.bias, 0.01)
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.01)
        torch.nn.init.zeros_(m.bias)

# ---------------------
# Disaggregation Model
# ---------------------
class DisAgg(nn.Module):
    
    """
    Super-Resolution of graphs
    
    """
    
    def __init__(self, low_size, high_size, hidden_sizes):
        super().__init__()
        
        self.low_size = low_size
        self.high_size = high_size
        self.hidden_size = hidden_sizes[0]
        
        ## lstm layers
        self.lstm0 = nn.LSTM(self.low_size, self.low_size, 1, batch_first=True)
        self.lstm1 = nn.LSTM(self.low_size, self.hidden_size, 1, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size, self.high_size, 1, batch_first=True)
        self.lstm3 = nn.LSTM(self.high_size, self.high_size, 1, batch_first=True)
        
    def forward(self, att):
        
        ## projection
        puma,_ = self.lstm0(att)
        nta,_ = self.lstm1(F.relu(puma))
        tract,_ = self.lstm2(F.relu(nta))
        tract,_ = self.lstm3(F.relu(tract))
        
        return puma, nta, tract