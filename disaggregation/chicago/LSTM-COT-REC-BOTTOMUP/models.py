import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
seed = 100

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
class DisAgg_com_tract(nn.Module):
    
    """
    Super-Resolution of graphs
    
    """
    
    def __init__(self, low_size, high_size, hidden_sizes, linkages):
        super().__init__()
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True

        self.low_size = low_size
        self.high_size = high_size
        self.hidden_size = hidden_sizes[0]
        self.com_tract = linkages[0]
        
        ## lstm layers
        self.lstm1 = nn.LSTM(self.low_size, self.low_size, 1, batch_first=True)
        self.lstm2 = nn.LSTM(self.low_size, self.high_size, 1, batch_first=True)
        self.lstm3 = nn.LSTM(self.high_size, self.high_size, 1, batch_first=True)
        
    def forward(self, att):
        
        ## projection
        com,_ = self.lstm1(att)
        tract,_ = self.lstm2(F.relu(com))
        tract,_ = self.lstm3(F.relu(tract))
        
        ## reconstruction
        rec_coms = [
            torch.matmul(tract, self.com_tract.T)
        ]
        
        return {'com': com,
                'tract': tract,
                'rec_coms':rec_coms}
    
# ---------------------
# Disaggregation Model
# ---------------------
class DisAgg_com_block(nn.Module):
    
    """
    Super-Resolution of graphs
    
    """
    
    def __init__(self, low_size, high_size, hidden_sizes, linkages):
        super().__init__()
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
        
        self.low_size = low_size
        self.high_size = high_size
        self.hidden_size = hidden_sizes[0]
        
        self.com_tract = linkages[0]
        self.com_block = linkages[1]
        self.tract_block = linkages[2]        
        
        ## lstm layers
        self.lstm0 = nn.LSTM(self.low_size, self.low_size, 1, batch_first=True)
        self.lstm1 = nn.LSTM(self.low_size, self.hidden_size, 1, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size, self.high_size, 1, batch_first=True)
        self.lstm3 = nn.LSTM(self.high_size, self.high_size, 1, batch_first=True)
        
    def forward(self, att):
        
        ## projection
        com,_ = self.lstm0(att)
        tract,_ = self.lstm1(F.relu(com))
        block,_ = self.lstm2(F.relu(tract))
        block,_ = self.lstm3(F.relu(block))
        
        ## reconstruction
        rec_coms = [
            torch.matmul(block, self.com_block.T),
        ]
        
        rec_tracts = [
            torch.matmul(block, self.tract_block.T)
        ]
        
        return {'com': com,
                'tract': tract, 
                'block':block,
                'rec_coms':rec_coms,
                'rec_tracts': rec_tracts}
    
# ---------------------
# Disaggregation Model
# ---------------------
class DisAgg_com_extreme(nn.Module):
    
    """
    Super-Resolution of graphs
    
    """
    
    def __init__(self, low_size, high_size, hidden_sizes, linkages):
        super().__init__()
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
        
        self.low_size = low_size
        self.high_size = high_size
        self.hidden_size_0 = hidden_sizes[0]
        self.hidden_size_1 = hidden_sizes[1]
        
        self.com_tract = linkages[0]
        self.com_block = linkages[1]
        self.com_extreme = linkages[2] 
        self.tract_block = linkages[3]
        self.tract_extreme = linkages[4]
        self.block_extreme = linkages[5]
        
        ## lstm layers
        self.lstm0 = nn.LSTM(self.low_size, self.low_size, 1, batch_first=True)
        self.lstm1 = nn.LSTM(self.low_size, self.hidden_size_0, 1, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size_0, self.hidden_size_1, 1, batch_first=True)
        self.lstm3 = nn.LSTM(self.hidden_size_1, self.high_size, 1, batch_first=True)
        self.lstm4 = nn.LSTM(self.high_size, self.high_size, 1, batch_first=True)
        
    def forward(self, att):
        
        ## projection
        com,_ = self.lstm0(att)
        tract,_ = self.lstm1(F.relu(com))
        block,_ = self.lstm2(F.relu(tract))
        extreme,_ = self.lstm3(F.relu(block))
        extreme,_ = self.lstm4(F.relu(extreme))
        
        ## reconstruction
        rec_coms = [
            torch.matmul(extreme, self.com_extreme.T)
        ]
        
        rec_tracts = [
            torch.matmul(extreme, self.tract_extreme.T)
        ]
        
        rec_blocks = [
            torch.matmul(extreme, self.block_extreme.T)
        ]
        
        return {'com': com,
                'tract': tract,
                'block':block,
                'extreme':extreme,                 
                'rec_coms':rec_coms,
                'rec_tracts': rec_tracts,
                'rec_blocks': rec_blocks}