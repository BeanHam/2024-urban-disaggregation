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
        self.hidden_size = hidden_sizes[0]
        
        self.puma_nta = linkages[0]
        self.puma_tract = linkages[1]
        self.nta_tract = linkages[2]        
        
        ## linear layers
        self.lr0 = nn.Linear(self.low_size, self.low_size)        
        self.lr1 = nn.Linear(self.low_size, self.hidden_size)
        self.lr2 = nn.Linear(self.hidden_size, self.high_size)
        self.lr3 = nn.Linear(self.high_size, self.high_size)
        
    def forward(self, att):
        
        ## projection
        puma = self.lr0(att)
        nta = self.lr1(F.relu(puma))
        tract = self.lr2(F.relu(nta))
        tract = self.lr3(F.relu(tract))
        
        ## reconstruction
        rec_pumas = [
            torch.matmul(nta, self.puma_nta.T),
            torch.matmul(tract, self.puma_tract.T),
        ]
        
        rec_ntas = [
            torch.matmul(tract, self.nta_tract.T)
        ]
        
        return {'puma': puma,
                'nta': nta, 
                'tract':tract,
                'rec_pumas':rec_pumas,
                'rec_ntas': rec_ntas}