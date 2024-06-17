import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"

# ---------------------------
# seeding for reproducibility
# ---------------------------
seed = 100
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True)
        
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
# min_max 3D normalization
# ---------------------        
def min_max_norm(e, exp_linkage):
    pos_vec = 9e15*torch.ones_like(e)
    neg_vec = -9e15*torch.ones_like(e)
    pos_e = torch.where(exp_linkage>0, e, pos_vec)
    neg_e = torch.where(exp_linkage>0, e, neg_vec)    
    min_values = torch.min(pos_e, dim=-1).values.unsqueeze_(-1)
    max_values = torch.max(neg_e, dim=-1).values.unsqueeze_(-1)
    denomenator = max_values-min_values
    ## when min and max values are the same
    denomenator = torch.where(denomenator==0, 1e-6, denomenator)
    e = (e-min_values)/denomenator
    e = torch.where(exp_linkage>0, e, neg_vec)
    return e 

# -----------------------------
# spatial diaggregation layer
# -----------------------------
class spatial_layer(nn.Module):
    
    """
    Super-Resolution of graphs
    
    """
    
    def __init__(self, low, high, local, linkage):
        super().__init__()
        
        ## parameters
        self.low = low
        self.high = high
        self.local = local
        self.linkage = linkage
        
        ## layers
        self.wk = nn.Linear(low, high)
        self.wq = nn.Linear(low, low)
        self.wv = nn.Linear(low, low)
        self.o_proj = nn.Linear(high, high)
        #self._reset_parameters()
    
    def _reset_parameters(self):
        
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.wk.bias.data.fill_(0)
        self.wq.bias.data.fill_(0)
        self.wv.bias.data.fill_(0)
        self.o_proj.bias.data.fill_(0)

    def forward(self, inputs):
        
        # inputs: (B,1,N)        
        ## global attention calculation
        ## -- global attention allows values of children regions to be from any parental region
        Q,K,V = self.wq(inputs), self.wk(inputs), self.wv(inputs)
        global_e = torch.matmul(Q.transpose(-1,-2), K)/np.sqrt(self.high)
        global_attention = F.softmax(global_e, dim=-1)
        x = torch.matmul(V, global_attention)

        ## local attention
        ## -- local attention only allows values of children regions to be from its connected parental region
        if self.local:
            batch_linkage = self.linkage.clone().unsqueeze_(0).repeat(inputs.size(0),1,1)
            local_e = min_max_norm(global_e, batch_linkage)
            local_attention = F.softmax(local_e, dim=-1)
            local_x = torch.matmul(V, local_attention)
            x = x + local_x
        #else:
        #    global_attention = F.softmax(global_e, dim=-1)
        #    x = torch.matmul(V, global_attention)
        
        ## output projection
        x = self.o_proj(x)

        return x

# -------------------------
# Spatial Disaggregtion
# -------------------------
class SpatialDisaggregtion(nn.Module):
    
    """
    multi-head attention layer
    """
    
    def __init__(self, low, high, nheads, local, linkage):
        
        super(SpatialDisaggregtion, self).__init__()
        
        self.attentions = [spatial_layer(low, high, local, linkage) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.o_proj = nn.Linear(high*nheads, high)        
        
    def forward(self, x):
        x = torch.cat([att(x) for att in self.attentions], dim=-1)
        x = self.o_proj(x)
        return x

# ---------------------
# GRUSPA cell
# ---------------------       
class GRUSPACell(nn.Module):

    """
    An implementation of GRUCell.

    """

    def __init__(self, low, high, nheads, local, linkage):
        super(GRUSPACell, self).__init__()
        
        self.low = low
        self.high = high
        self.x2r = SpatialDisaggregtion(low, high, nheads, local, linkage)
        self.x2z = SpatialDisaggregtion(low, high, nheads, local, linkage)
        self.x2n = SpatialDisaggregtion(low, high, nheads, local, linkage)        
        self.h2r = nn.Linear(high, high, bias=True)
        self.h2z = nn.Linear(high, high, bias=True)
        self.h2n = nn.Linear(high, high, bias=True)
        #self.reset_parameters()

    def _reset_parameters(self):
        
        nn.init.xavier_uniform_(self.h2r.weight)
        nn.init.xavier_uniform_(self.h2r.weight)
        nn.init.xavier_uniform_(self.h2r.weight)
        self.h2r.bias.data.fill_(0)
        self.h2r.bias.data.fill_(0)
        self.h2r.bias.data.fill_(0)
    
    def forward(self, x, h):
        
        ## reset gate
        x2r = self.x2r(x)
        h2r = self.h2r(h)
        resetgate = torch.sigmoid(x2r + h2r)
        
        ## update gate
        x2z = self.x2z(x)
        h2z = self.h2z(h)
        updatetgate = torch.sigmoid(x2z + h2z)
        
        ## new gate
        x2n = self.x2n(x)
        h2n = self.h2n(h*resetgate)
        newgate = torch.tanh(x2n + h2n)
        
        ## new hidden state        
        hy = updatetgate*h + (1-updatetgate)*newgate
        
        return hy

# ---------------------
# STTD model
# ---------------------      
class GRUSPA(nn.Module):
    def __init__(self, 
                 linkage,
                 parameters):
        super(GRUSPA, self).__init__()
        
        # parameters
        self.parameters = parameters
        self.low = parameters['low']
        self.high = parameters['high']
        self.num_heads = parameters['num_heads']
        self.time_steps = parameters['time_steps']
        self.local = parameters['local']=='yes'
        self.gruspa_cell = GRUSPACell(self.low, 
                                      self.high,
                                      self.num_heads,
                                      self.local,
                                      linkage)            
             
    def forward(self, x):
        
        # x: (batch_size, in_steps, num_nodes)
        hn = Variable(torch.zeros(x.size(0), 1, self.high).cuda())
        out = []
        for seq in range(x.size(1)):
            hn = self.gruspa_cell(x[:,seq,:].clone().unsqueeze_(1), hn) 
            out.append(hn)
        out = torch.concat(out,dim=1)
        return out     
    
# ---------------------
# PUMA TO NTA
# ---------------------
class puma_nta(nn.Module):
    
    """
    Super-Resolution of graphs
    
    """
    
    def __init__(self, linkage, rec, parameters):
        super().__init__()
        
        self.rec = rec
        self.linkage = linkage
        self.gruspa = GRUSPA(linkage, parameters)

    def forward(self, x):
        
        ## projection
        nta = self.gruspa(x)

        ## reconstruction
        if self.rec == 'no':
            rec_puma = None
        else: 
            rec_puma = torch.matmul(nta, self.linkage.T)
        
        return {
            'nta': nta,
            'rec_puma':rec_puma
        }

# ---------------------
# PUMA TO TRACT
# ---------------------    
class puma_tract(nn.Module):
    
    """
    Super-Resolution of graphs
    
    """
    
    def __init__(self, linkage, rec, parameters):
        super().__init__()
        
        self.rec = rec        
        self.linkage = linkage
        self.gruspa = GRUSPA(linkage, parameters)

    def forward(self, x):
        
        ## projection
        tract = self.gruspa(x)

        ## reconstruction
        if self.rec == 'no': 
            rec_puma = None
        else:
            rec_puma = torch.matmul(tract, self.linkage.T)
            
        return {
            'tract': tract,
            'rec_puma':rec_puma
        }

# ---------------------
# PUMA TO BLOCK
# ---------------------    
class puma_block(nn.Module):
    
    """
    Super-Resolution of graphs
    
    """
    
    def __init__(self, linkage, rec, parameters):
        super().__init__()
        
        self.rec = rec        
        self.linkage = linkage
        self.gruspa = GRUSPA(linkage, parameters)

    def forward(self, x):
        
        ## projection
        block = self.gruspa(x)

        ## reconstruction
        if self.rec == 'no': 
            rec_puma = None
        else:
            rec_puma = torch.matmul(block, self.linkage.T)
        
        return {
            'block': block,
            'rec_puma':rec_puma
        }    
