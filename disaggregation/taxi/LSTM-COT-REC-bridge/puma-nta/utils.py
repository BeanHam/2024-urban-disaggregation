import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------------
# Prepare Data
# ---------------------
class taxi_data(torch.utils.data.Dataset):
    
    """
    Prepare taxi data
    """
    
    def __init__(self, 
                 source_low, 
                 source_high,
                 chunk_size):
        super(taxi_data, self).__init__()
        
        self.source_low = source_low
        self.source_high = source_high
        self.chunk_size = chunk_size
        
    def __getitem__(self, index):
        
        ## batch attributes
        max_index = np.min([len(self.source_low), index+self.chunk_size])
        batch_source_low = self.source_low[index:max_index]
        batch_source_high = self.source_high[index:max_index]
        
        return batch_source_low, batch_source_high

    def __len__(self):
        return len(self.source_low)

# ---------------------
# Load Data
# ---------------------
def load_data(low, high, parameters):
    
    """
    Function to load datasets
    
    Arg:
        - parameters: parameter json file
    """
    
    ## data path
    chunk_size = parameters['chunk_size']
    source_path = parameters['source_path']
    source_low_path = source_path+'/attributes/'+low
    source_high_path = source_path+'/attributes/'+high
    puma_nta_linkage_path = source_path+'/linkages/puma_nta.npy'
    
    ## load data
    X_low_train = torch.from_numpy(np.load(source_low_path+'_train.npy')).float()
    X_low_val = torch.from_numpy(np.load(source_low_path+'_val.npy')).float()
    X_low_test = torch.from_numpy(np.load(source_low_path+'_test.npy')).float()
    X_high_train = torch.from_numpy(np.load(source_high_path+'_train.npy')).float()
    X_high_val = torch.from_numpy(np.load(source_high_path+'_val.npy')).float()
    X_high_test = torch.from_numpy(np.load(source_high_path+'_test.npy')).float()
    
    ## linkages
    puma_nta_linkage = torch.from_numpy(np.load(puma_nta_linkage_path)).float()
    linkages = [puma_nta_linkage]
    
    ## min-max normalization: min=0
    X_low_max = np.max(
        [
            torch.max(X_low_train).item(),
            torch.max(X_low_val).item(),
            torch.max(X_low_test).item(),]
    )
    X_low_train = (X_low_train/X_low_max)[:,:,None]
    X_low_val = (X_low_val/X_low_max)[:,:,None]
    X_low_test = (X_low_test/X_low_max)[:,:,None]
    X_high_train = (X_high_train/X_low_max)[:,:,None]
    X_high_val = (X_high_val/X_low_max)[:,:,None]
    X_high_test = (X_high_test/X_low_max)[:,:,None]
    
    ## prepare data
    dataset_train = taxi_data(X_low_train, X_high_train, chunk_size)
    dataset_val = taxi_data(X_low_val, X_high_val, chunk_size)
    dataset_test = taxi_data(X_low_test, X_high_test, chunk_size)
    
    return dataset_train, dataset_val, dataset_test, X_low_max, linkages

# ---------------------
# Training Function
# ---------------------
def train(model, 
          criterion, 
          optimizer,
          device,
          batch_size, 
          dataset,
          parameters):
    
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
    
    ## iterate through training dataset
    chunk_size = parameters['chunk_size']
    for i in range(0, len(dataset)-chunk_size, batch_size):
        
        ## batch data
        indices = range(i, min(len(dataset)-chunk_size, i+batch_size))      
        source_low, source_high = zip(*[dataset[j] for j in indices])
        source_low = torch.stack(source_low).squeeze_(-1).to(device)
        source_high = torch.stack(source_high).squeeze_(-1).to(device)
        
        ## prediction
        results = model(source_low)
        puma = results['puma']
        nta = results['nta']
        rec_pumas = results['rec_pumas']
        
        ## loss        
        loss_puma = torch.sum(torch.abs(puma-source_low))/puma.size(0)/chunk_size/3751
        loss_nta = torch.sum(torch.abs(nta-source_high))/puma.size(0)/chunk_size/3751
        
        ## reconstruction loss
        for rec_puma in rec_pumas:
            loss_puma+=torch.sum(torch.abs(rec_puma-source_low))/puma.size(0)/chunk_size/3751
        loss_puma = loss_puma/(len(rec_pumas)+1)        
        
        ## total loss
        loss = 10.0*loss_puma + 8.5*loss_nta
        
        ## back propogration
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
# ---------------------
# Evalaution Function
# ---------------------         
def evaluation(model, 
               criterion, 
               device,
               batch_size, 
               dataset,
               parameters):
    
        
    """
    Function to evaluate the model
    
    Arg:
        - model
        - criterion: loss function
        - batch_size
        - dataset: validation/test dataset
        
    """
    
    ## storages
    pred_puma = []
    pred_nta = []
    rec_puma = []
    chunk_size = parameters['chunk_size']
    outer_indices = [k for k in range(0, len(dataset)-chunk_size+1, chunk_size)]
    
    ## iterate through evaluation dataset
    for i in range(0, len(outer_indices), batch_size):
        
        ## batch data
        inner_indices = range(i, min(len(outer_indices), i+batch_size))
        source_low, _ = zip(*[dataset[outer_indices[j]] for j in inner_indices])
        source_low = torch.stack(source_low).squeeze_(-1).to(device)
        
        ## prediction
        with torch.no_grad():
            results = model(source_low)       
            puma = results['puma']
            nta = results['nta']
            rec_pumas = results['rec_pumas']
        pred_puma.append(puma)
        pred_nta.append(nta)
        rec_puma.append(rec_pumas)
    
    ## aggregate
    pred_puma = torch.cat(pred_puma).cpu()    
    pred_nta = torch.cat(pred_nta).cpu()
    rec_puma = torch.cat([torch.stack(r) for r in rec_puma],dim=1).cpu()    
    pred_puma = pred_puma.reshape(-1, pred_puma.size(-1))
    pred_nta = pred_nta.reshape(-1, pred_nta.size(-1))
    rec_puma = rec_puma.reshape(rec_puma.size(0), -1, rec_puma.size(-1))    
    gt_puma = dataset.source_low.squeeze_(-1)[:len(pred_puma)]
    gt_nta = dataset.source_high.squeeze_(-1)[:len(pred_nta)]
    
    ## calculate loss
    loss_puma = torch.sum(torch.abs(pred_puma-gt_puma)).item()/pred_puma.size(0)/3751
    loss_nta = torch.sum(torch.abs(pred_nta-gt_nta)).item()/pred_nta.size(0)/3751
    
    ## reconstruction loss
    loss_puma_rec = 0
    for r in rec_puma:
        loss_puma_rec+=torch.sum(torch.abs(r-gt_puma)).item()/pred_puma.size(0)/3751
    loss_puma_rec = loss_puma_rec/(len(rec_puma))
    
    return {'loss_puma': loss_puma,
            'loss_nta': loss_nta,
            'loss_puma_rec': loss_puma_rec,            
            'gt_nta': gt_nta,
            'pred_nta': pred_nta}

# ---------------------
# Early Stop Function
# ---------------------
class EarlyStopping():
    def __init__(self, 
                 model,
                 tolerance=20):

        self.model = model
        self.tolerance = tolerance
        self.loss_min = np.inf
        self.counter = 0
        self.early_stop = False
        self.save_model = False
        
    def __call__(self, loss):
        if loss > self.loss_min:
            self.counter +=1
            self.save_model = False
            if self.counter >= self.tolerance:  
                self.early_stop = True
        else:
            self.save_model = True
            self.loss_min = loss
            self.counter = 0 