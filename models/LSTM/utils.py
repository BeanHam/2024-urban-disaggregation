import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class taxi_data(torch.utils.data.Dataset):
    
    """
    Prepare taxi data
    """
    
    def __init__(self, 
                 att_low, 
                 att_super, 
                 adj_low,
                 adj_super,
                 linkage,
                 chunk_size):
        super(taxi_data, self).__init__()
        self.att_low = att_low
        self.att_super = att_super
        self.adj_low = adj_low
        self.adj_super = adj_super
        self.linkage = linkage
        self.chunk_size = chunk_size
        
    def __getitem__(self, index):
        
        ## batch attributes
        batch_att_low = self.att_low[index:index+self.chunk_size]
        batch_att_super = self.att_super[index:index+self.chunk_size]
        
        return batch_att_low, batch_att_super, self.adj_low, self.adj_super, self.linkage

    def __len__(self):
        return len(self.att_low)
    


def load_data(att_low_res_path, 
              adj_low_res_path, 
              att_super_res_path, 
              adj_super_res_path,
              linkage_path,
              chunk_size):
    
    """
    Function to load datasets
    
    Arg:
        - path: path to node attributes & adjacency matrix
    """
    
    ## load data
    X_low = torch.from_numpy(np.load(att_low_res_path)).float()
    X_super = torch.from_numpy(np.load(att_super_res_path)).float()
    A_low = torch.from_numpy(np.load(adj_low_res_path)).float()
    A_super = torch.from_numpy(np.load(adj_super_res_path)).float()
    linkage = torch.from_numpy(np.load(linkage_path)).float()
    #linkage = linkage/torch.sum(linkage,dim=1, keepdim=True) ## normalize
    
    ## min-max normalization: min=0
    X_low = X_low/torch.max(X_low)
    X_low = X_low[:,:,None]
    X_super = X_super/torch.max(X_super)
    X_super = X_super[:,:,None]
    
    ## new adjacency matrix
    D1 = torch.diag(torch.sum(A_low,dim=1)).float()
    D2 = torch.diag(torch.sum(A_super,dim=1)).float()
    D1_tilda = torch.sqrt(torch.linalg.inv(D1))
    D2_tilda = torch.sqrt(torch.linalg.inv(D2))
    A_low = D1_tilda@A_low@D1_tilda
    A_super = D2_tilda@A_super@D2_tilda
    
    ## split train, val & test
    X_low_train, X_low_test, X_super_train, X_super_test = train_test_split(X_low, 
                                                                            X_super, 
                                                                            test_size=0.2, 
                                                                            random_state=1)
    X_low_train, X_low_val, X_super_train, X_super_val = train_test_split(X_low_train, 
                                                                          X_super_train, 
                                                                          test_size=0.1, 
                                                                          random_state=1)
    ## prepare data
    dataset_train = taxi_data(X_low_train, X_super_train, A_low, A_super, linkage, chunk_size)
    dataset_val = taxi_data(X_low_val, X_super_val, A_low, A_super, linkage, chunk_size)
    dataset_test = taxi_data(X_low_test, X_super_test, A_low, A_super, linkage, chunk_size)
    
    return dataset_train, dataset_val, dataset_test


def train(dataset,
          model, 
          criterion, 
          optimizer, 
          batch_size, 
          chunk_size,
          device):
    
    """
    Function to train the model
    
    Arg:
        - model
        - criterion: loss function
        - optimizer
        - scheduler: learing rate updater
        - batch_size
        - dataset: training dataset
        
    """
    
    ## training  
    ## iterate through training dataset
    for i in range(0, len(dataset)-chunk_size+1, batch_size):
        
        ## batch data
        indices = range(i, min(len(dataset)-chunk_size+1, i+batch_size))        
        X_batch_low, X_batch_super, A_low, A_super, linkage = zip(*[dataset[i] for i in indices])
        X_batch_low = torch.stack(X_batch_low).squeeze(-1).to(device)
        X_batch_super = torch.stack(X_batch_super).squeeze(-1).to(device)
        A_low = A_low[0].to(device)
        A_super = A_super[0].to(device)
        linkage = linkage[0].to(device)
        
        ## prediction
        pred,_,_ = model(X_batch_low)
        
        ## loss
        loss = criterion(X_batch_super, pred)
        
        ## back propogration
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
def evaluation(dataset,
               model, 
               criterion, 
               batch_size,
               chunk_size,
               device):
    
        
    """
    Function to evaluate the model
    
    Arg:
        - model
        - criterion: loss function
        - batch_size
        - dataset: validation/test dataset
        
    """
    
    ## evaluation
    pred_super = []
    gt_super = []
    gt_low = []
    
    for i in range(0, len(dataset)-chunk_size+1, batch_size):
        indices = range(i, min(len(dataset)-chunk_size+1, i+batch_size))        
        X_batch_low, X_batch_super, A_low, A_super, linkage = zip(*[dataset[i] for i in indices])
        X_batch_low = torch.stack(X_batch_low).squeeze(-1).to(device)
        X_batch_super = torch.stack(X_batch_super).squeeze(-1).to(device)
        A_low = A_low[0].to(device)
        A_super = A_super[0].to(device)
        linkage = linkage[0].to(device)
        
        with torch.no_grad():
            pred,_,_ = model(X_batch_low)
            
        pred_super.append(pred[:,-1,:])
        gt_super.append(X_batch_super[:,-1,:])
        gt_low.append(X_batch_low[:,-1,:])
    
    pred_super = torch.cat(pred_super)
    gt_super = torch.cat(gt_super)
    gt_low = torch.cat(gt_low)
    loss = criterion(pred_super, gt_super).cpu().item()
    
    return loss, pred_super, gt_super, gt_low
    