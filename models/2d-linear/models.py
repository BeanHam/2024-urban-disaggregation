import torch
import torch.nn as nn
import torch.nn.functional as F

## seed
seed = 816
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class GraphSR(nn.Module):
    
    """
    Super-Resolution of graphs
    
    """
    
    def __init__(self, linkage):
        super().__init__()
        
        self.linkage = linkage
                
        ## projection layer
        self.proj = nn.Parameter(torch.rand_like(self.linkage))
        self.proj_bias = nn.Parameter(torch.rand(self.linkage.size(1),1))

        ## initialization
        nn.init.xavier_uniform_(self.proj.data, gain=0.01)
        nn.init.constant_(self.proj_bias.data, 0.01)

    def forward(self, att):
        
        ## projection layers
        x = F.relu((self.proj).T@att + self.proj_bias)
        return x
    
