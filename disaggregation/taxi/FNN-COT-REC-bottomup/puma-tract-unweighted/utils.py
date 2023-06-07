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
                 source_nta):
        super(taxi_data, self).__init__()
        
        self.source_low = source_low
        self.source_high = source_high
        self.source_nta = source_nta
        
    def __getitem__(self, index):
        
        ## batch attributes
        batch_source_low = self.source_low[index]
        batch_source_high = self.source_high[index]
        batch_source_nta = self.source_nta[index]
        
        return batch_source_low, batch_source_high, batch_source_nta

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
    source_path = parameters['source_path']
    source_low_path = source_path+'/attributes/'+low
    source_high_path = source_path+'/attributes/'+high
    source_nta_path = source_path+'/attributes/nta'      
    puma_nta_linkage_path = source_path+'/linkages/puma_nta.npy'
    puma_tract_linkage_path = source_path+'/linkages/puma_tract.npy'
    nta_tract_linkage_path = source_path+'/linkages/nta_tract.npy'
           
    ## load data
    X_low_train = torch.from_numpy(np.load(source_low_path+'_train.npy')).float()
    X_low_val = torch.from_numpy(np.load(source_low_path+'_val.npy')).float()
    X_low_test = torch.from_numpy(np.load(source_low_path+'_test.npy')).float()
    X_high_train = torch.from_numpy(np.load(source_high_path+'_train.npy')).float()
    X_high_val = torch.from_numpy(np.load(source_high_path+'_val.npy')).float()
    X_high_test = torch.from_numpy(np.load(source_high_path+'_test.npy')).float()
    X_nta_train = torch.from_numpy(np.load(source_nta_path+'_train.npy')).float()
    X_nta_val = torch.from_numpy(np.load(source_nta_path+'_val.npy')).float()
    X_nta_test = torch.from_numpy(np.load(source_nta_path+'_test.npy')).float()
    
    ## linkages
    puma_nta_linkage = torch.from_numpy(np.load(puma_nta_linkage_path)).float()
    puma_tract_linkage = torch.from_numpy(np.load(puma_tract_linkage_path)).float()
    nta_tract_linkage = torch.from_numpy(np.load(nta_tract_linkage_path)).float()
    linkages = [puma_nta_linkage, puma_tract_linkage, nta_tract_linkage]
    
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
    X_nta_train = (X_nta_train/X_low_max)[:,:,None]
    X_nta_val = (X_nta_val/X_low_max)[:,:,None]
    X_nta_test = (X_nta_test/X_low_max)[:,:,None]
            
    ## prepare data
    dataset_train = taxi_data(X_low_train, X_high_train, X_nta_train)
    dataset_val = taxi_data(X_low_val, X_high_val, X_nta_val)
    dataset_test = taxi_data(X_low_test, X_high_test, X_nta_test)
    
    return dataset_train, dataset_val, dataset_test, X_low_max, linkages

# ---------------------
# Training Function
# ---------------------
def train(model, 
          criterion, 
          optimizer,
          device,
          batch_size, 
          dataset):
    
    """
    Function to train the model
    
    Arg:
        - model
        - criterion: loss function
        - optimizer
        - dataset: training dataset
        
    """
    
    ## iterate through training dataset
    for i in range(0, len(dataset), batch_size):
        
        ## batch data
        indices = range(i, min(len(dataset), i+batch_size))        
        source_low, source_high, source_nta = zip(*[dataset[i] for i in indices])
        source_low = torch.stack(source_low).squeeze_(-1).to(device)
        source_high = torch.stack(source_high).squeeze_(-1).to(device)
        source_nta = torch.stack(source_nta).squeeze_(-1).to(device)
        
        ## prediction
        results = model(source_low)
        puma = results['puma']
        nta = results['nta']
        tract = results['tract']        
        rec_pumas = results['rec_pumas']
        rec_ntas = results['rec_ntas']        
        
        ## loss
        loss_puma = torch.sum(torch.abs(puma-source_low))/puma.size(0)/3751
        loss_nta = torch.sum(torch.abs(nta-source_nta))/puma.size(0)/3751
        loss_tract = torch.sum(torch.abs(tract-source_high))/puma.size(0)/3751
        
        ## reconstruction loss
        for rec_puma in rec_pumas:
            loss_puma+=torch.sum(torch.abs(rec_puma-source_low))/puma.size(0)/3751
        loss_puma = loss_puma/(len(rec_pumas)+1)
        
        for rec_nta in rec_ntas:
            loss_nta+=torch.sum(torch.abs(rec_nta-source_nta))/puma.size(0)/3751
        loss_nta = loss_nta/(len(rec_ntas)+1)
        
        loss = loss_puma + loss_nta + loss_tract
        
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
               dataset):
    
        
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
    pred_tract = [] 
    rec_puma = []
    rec_nta = []
    
    ## iterate through evaluation dataset
    for i in range(0, len(dataset), batch_size):
        
        ## batch data
        indices = range(i, min(len(dataset), i+batch_size))        
        source_low, source_high, source_nta = zip(*[dataset[i] for i in indices])
        source_low = torch.stack(source_low).squeeze_(-1).to(device)
        source_high = torch.stack(source_high).squeeze_(-1).to(device)
        source_nta = torch.stack(source_nta).squeeze_(-1).to(device)
        
        ## prediction
        with torch.no_grad():
            results = model(source_low)       
            puma = results['puma']
            nta = results['nta']
            tract = results['tract']
            rec_pumas = results['rec_pumas']
            rec_ntas = results['rec_ntas']
        pred_puma.append(puma)
        pred_nta.append(nta)
        pred_tract.append(tract)
        rec_puma.append(rec_pumas)
        rec_nta.append(rec_ntas)
        
    ## aggregate
    pred_puma = torch.cat(pred_puma).squeeze_(1).cpu()
    pred_nta = torch.cat(pred_nta).squeeze_(1).cpu()
    pred_tract = torch.cat(pred_tract).squeeze_(1).cpu()    
    rec_puma = torch.cat([torch.stack(r) for r in rec_puma],dim=1).cpu()
    rec_nta = torch.cat([torch.stack(r) for r in rec_nta],dim=1).cpu()    
    gt_puma = dataset.source_low.squeeze_(-1)
    gt_nta = dataset.source_nta.squeeze_(-1)
    gt_tract = dataset.source_high.squeeze_(-1)
    
    ## calculate loss
    loss_puma = torch.sum(torch.abs(pred_puma-gt_puma)).item()/pred_puma.size(0)/3751
    loss_nta = torch.sum(torch.abs(pred_nta-gt_nta)).item()/pred_puma.size(0)/3751
    loss_tract = torch.sum(torch.abs(pred_tract-gt_tract)).item()/pred_puma.size(0)/3751
    
    ## reconstruction loss
    loss_puma_rec = 0
    for r in rec_puma:
        loss_puma_rec+=torch.sum(torch.abs(r-gt_puma)).item()/pred_puma.size(0)/3751
    loss_puma_rec = loss_puma_rec/(len(rec_puma))
    
    loss_nta_rec = 0
    for r in rec_nta:
        loss_nta_rec+=torch.sum(torch.abs(r-gt_nta)).item()/pred_puma.size(0)/3751
    loss_nta_rec = loss_nta_rec/(len(rec_nta))  
    
    return {'loss_puma': loss_puma,
            'loss_nta': loss_nta,
            'loss_tract': loss_tract,
            'loss_puma_rec': loss_puma_rec,
            'loss_nta_rec': loss_nta_rec,
            'gt_tract': gt_tract,
            'pred_tract': pred_tract}

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