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
    
    def __init__(self, low_size, high_size, hidden_sizes, linkages):
        super().__init__()
        
        self.low_size = low_size
        self.high_size = high_size
        self.hidden_size_0 = hidden_sizes[0]
        self.hidden_size_1 = hidden_sizes[1]
        
        self.puma_nta = linkages[0]
        self.puma_tract = linkages[1]
        self.puma_block = linkages[2] 
        self.nta_tract = linkages[3]
        self.nta_block = linkages[4]
        self.tract_block = linkages[5]
        
        ## lstm layers
        self.lstm0 = nn.LSTM(self.low_size, self.low_size, 1, batch_first=True)
        self.lstm1 = nn.LSTM(self.low_size, self.hidden_size_0, 1, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size_0, self.hidden_size_1, 1, batch_first=True)
        self.lstm3 = nn.LSTM(self.hidden_size_1, self.high_size, 1, batch_first=True)
        self.lstm4 = nn.LSTM(self.high_size, self.high_size, 1, batch_first=True)
        
    def forward(self, att):
        
        ## projection
        puma,_ = self.lstm0(att)
        nta,_ = self.lstm1(F.relu(puma))
        tract,_ = self.lstm2(F.relu(nta))
        block,_ = self.lstm3(F.relu(tract))
        block,_ = self.lstm4(F.relu(block))
        
        ## reconstruction
        rec_pumas = [
            torch.matmul(nta, self.puma_nta.T),
            torch.matmul(tract, self.puma_tract.T),
            torch.matmul(block, self.puma_block.T)
        ]
        
        rec_ntas = [
            torch.matmul(tract, self.nta_tract.T),
            torch.matmul(block, self.nta_block.T)
        ]
        
        rec_tracts = [
            torch.matmul(block, self.tract_block.T)
        ]
        
        return {'puma': puma,
                'nta': nta,
                'tract':tract,
                'block':block,                 
                'rec_pumas':rec_pumas,
                'rec_ntas': rec_ntas,
                'rec_tracts': rec_tracts}