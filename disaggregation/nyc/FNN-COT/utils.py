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
                 source_puma, 
                 source_nta,
                 source_tract,
                 source_block,
                 source_extreme):
        super(taxi_data, self).__init__()
        
        self.source_puma = source_puma
        self.source_nta = source_nta
        self.source_tract = source_tract
        self.source_block = source_block
        self.source_extreme = source_extreme
        
    def __getitem__(self, index):
        
        ## batch attributes
        batch_source_puma = self.source_puma[index]
        batch_source_nta = self.source_nta[index]
        batch_source_tract = self.source_tract[index]
        batch_source_block = self.source_block[index]
        batch_source_extreme = self.source_extreme[index]
        
        return batch_source_puma, \
               batch_source_nta, \
               batch_source_tract, \
               batch_source_block, \
               batch_source_extreme

    def __len__(self):
        return len(self.source_puma)

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
    source_puma_path = source_path+'/attributes/puma'    
    source_nta_path = source_path+'/attributes/nta'
    source_tract_path = source_path+'/attributes/tract'
    source_block_path = source_path+'/attributes/block'
    source_extreme_path = source_path+'/attributes/extreme'
    
    ## load data
    X_puma_train = torch.from_numpy(np.load(source_puma_path+'_train.npy')).float()
    X_puma_val = torch.from_numpy(np.load(source_puma_path+'_val.npy')).float()
    X_puma_test = torch.from_numpy(np.load(source_puma_path+'_test.npy')).float()
        
    X_nta_train = torch.from_numpy(np.load(source_nta_path+'_train.npy')).float()
    X_nta_val = torch.from_numpy(np.load(source_nta_path+'_val.npy')).float()
    X_nta_test = torch.from_numpy(np.load(source_nta_path+'_test.npy')).float()    
    
    X_tract_train = torch.from_numpy(np.load(source_tract_path+'_train.npy')).float()
    X_tract_val = torch.from_numpy(np.load(source_tract_path+'_val.npy')).float()
    X_tract_test = torch.from_numpy(np.load(source_tract_path+'_test.npy')).float()
    
    X_block_train = torch.from_numpy(np.load(source_block_path+'_train.npy')).float()
    X_block_val = torch.from_numpy(np.load(source_block_path+'_val.npy')).float()
    X_block_test = torch.from_numpy(np.load(source_block_path+'_test.npy')).float()
    
    X_extreme_train = torch.from_numpy(np.load(source_extreme_path+'_train.npy')).float()
    X_extreme_val = torch.from_numpy(np.load(source_extreme_path+'_val.npy')).float()
    X_extreme_test = torch.from_numpy(np.load(source_extreme_path+'_test.npy')).float()
    
    ## min-max normalization: min=0
    X_low_max = np.max(
        [
            torch.max(X_puma_train).item(),
            torch.max(X_puma_val).item(),
            torch.max(X_puma_test).item(),]
    )
    X_puma_train = (X_puma_train/X_low_max)[:,:,None]
    X_puma_val = (X_puma_val/X_low_max)[:,:,None]
    X_puma_test = (X_puma_test/X_low_max)[:,:,None]
    X_nta_train = (X_nta_train/X_low_max)[:,:,None]
    X_nta_val = (X_nta_val/X_low_max)[:,:,None]
    X_nta_test = (X_nta_test/X_low_max)[:,:,None]
    X_tract_train = (X_tract_train/X_low_max)[:,:,None]
    X_tract_val = (X_tract_val/X_low_max)[:,:,None]
    X_tract_test = (X_tract_test/X_low_max)[:,:,None]
    X_block_train = (X_block_train/X_low_max)[:,:,None]
    X_block_val = (X_block_val/X_low_max)[:,:,None]
    X_block_test = (X_block_test/X_low_max)[:,:,None]
    X_extreme_train = (X_extreme_train/X_low_max)[:,:,None]
    X_extreme_val = (X_extreme_val/X_low_max)[:,:,None]
    X_extreme_test = (X_extreme_test/X_low_max)[:,:,None]
    
    ## prepare data
    dataset_train = taxi_data(X_puma_train, X_nta_train, X_tract_train, X_block_train, X_extreme_train)
    dataset_val = taxi_data(X_puma_val, X_nta_val, X_tract_val, X_block_val, X_extreme_val)
    dataset_test = taxi_data(X_puma_test, X_nta_test, X_tract_test, X_block_test, X_extreme_test)
    
    return dataset_train, dataset_val, dataset_test, X_low_max
        
# ---------------------
# Training Function
# ---------------------
def train(model,
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
        - dataset: training dataset
        
    """
    
    ## iterate through training dataset
    dims = parameters['dims']
    high = parameters['high']
    for i in range(0, len(dataset), batch_size):
        
        ## batch data
        indices = range(i, min(len(dataset), i+batch_size))        
        source_puma, source_nta, source_tract, source_block, source_extreme = zip(*[dataset[j] for j in indices])
        source_puma = torch.stack(source_puma).squeeze_(-1).to(device)
        source_nta = torch.stack(source_nta).squeeze_(-1).to(device)
        source_tract = torch.stack(source_tract).squeeze_(-1).to(device)
        source_block = torch.stack(source_block).squeeze_(-1).to(device)
        source_extreme = torch.stack(source_extreme).squeeze_(-1).to(device)        
        
        ## prediction
        results = model(source_puma)
        
        if high == 'nta':
            puma = results['puma']
            nta = results['nta']       
            loss_puma = torch.mean(torch.sum(torch.abs(puma-source_puma), dim=1)/dims['block'])
            loss_nta = torch.mean(torch.sum(torch.abs(nta-source_nta), dim=1)/dims['block'])       
            loss = 10.0*loss_puma + 8.5*loss_nta
        elif high == 'tract':
            puma = results['puma']
            nta = results['nta']
            tract = results['tract']          
            loss_puma = torch.mean(torch.sum(torch.abs(puma-source_puma), dim=1)/dims['block'])
            loss_nta = torch.mean(torch.sum(torch.abs(nta-source_nta), dim=1)/dims['block'])
            loss_tract = torch.mean(torch.sum(torch.abs(tract-source_tract), dim=1)/dims['block'])
            loss = 10.0*loss_puma + 8.5*loss_nta + 5.0*loss_tract
        elif high == 'block':
            puma = results['puma']
            nta = results['nta']
            tract = results['tract']
            block = results['block']
            loss_puma = torch.mean(torch.sum(torch.abs(puma-source_puma), dim=1)/dims['block'])
            loss_nta = torch.mean(torch.sum(torch.abs(nta-source_nta), dim=1)/dims['block'])
            loss_tract = torch.mean(torch.sum(torch.abs(tract-source_tract), dim=1)/dims['block'])
            loss_block = torch.mean(torch.sum(torch.abs(block-source_block), dim=1)/dims['block'])         
            loss = 10.0*loss_puma + 8.5*loss_nta + 5.0*loss_tract + 1.5*loss_block
        else: 
            puma = results['puma']
            nta = results['nta']
            tract = results['tract']
            block = results['block']
            extreme = results['extreme']     
            loss_puma = torch.mean(torch.sum(torch.abs(puma-source_puma), dim=1)/dims['block'])
            loss_nta = torch.mean(torch.sum(torch.abs(nta-source_nta), dim=1)/dims['block'])
            loss_tract = torch.mean(torch.sum(torch.abs(tract-source_tract), dim=1)/dims['block'])
            loss_block = torch.mean(torch.sum(torch.abs(block-source_block), dim=1)/dims['block'])
            loss_extreme = torch.mean(torch.sum(torch.abs(extreme-source_extreme), dim=1)/dims['block'])                   
            loss = 10.0*loss_puma + 8.5*loss_nta + 5.0*loss_tract + 1.5*loss_block + loss_extreme
        
        ## back propogration
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
# ---------------------
# Evalaution Function
# ---------------------         
def evaluation(model, 
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
    pred = []
    dims = parameters['dims']
    high = parameters['high']
    
    ## iterate through evaluation dataset
    for i in range(0, len(dataset), batch_size):
        
        ## batch data
        indices = range(i, min(len(dataset), i+batch_size))
        source_puma, source_nta, source_tract, source_block, source_extreme = zip(*[dataset[j] for j in indices])
        source_puma = torch.stack(source_puma).squeeze_(-1).to(device)
        source_nta = torch.stack(source_nta).squeeze_(-1).to(device)
        source_tract = torch.stack(source_tract).squeeze_(-1).to(device)
        source_block = torch.stack(source_block).squeeze_(-1).to(device)
        source_extreme = torch.stack(source_extreme).squeeze_(-1).to(device)
        
        ## prediction
        with torch.no_grad():
            results = model(source_puma)
            
        if high == 'nta':
            nta = results['nta']
            pred.append(nta)
        elif high == 'tract':
            tract = results['tract']
            pred.append(tract)
        elif high == 'block':
            block = results['block']
            pred.append(block)
        else: 
            extreme = results['extreme']
            pred.append(extreme)
        
    ## aggregate
    if high == 'nta':
        pred = torch.cat(pred).squeeze_(1).cpu()
        gt = dataset.source_nta.squeeze_(-1)                
        loss = torch.mean(torch.sum(torch.abs(pred-gt), dim=1)/dims['block']).item()
    elif high == 'tract':
        pred = torch.cat(pred).squeeze_(1).cpu()
        gt = dataset.source_tract.squeeze_(-1)
        loss = torch.mean(torch.sum(torch.abs(pred-gt), dim=1)/dims['block']).item()
    elif high == 'block':
        pred = torch.cat(pred).squeeze_(1).cpu()        
        gt = dataset.source_block.squeeze_(-1)
        loss = torch.mean(torch.sum(torch.abs(pred-gt), dim=1)/dims['block']).item()
    else:
        pred = torch.cat(pred).squeeze_(1).cpu()        
        gt = dataset.source_extreme.squeeze_(-1)
        loss = torch.mean(torch.sum(torch.abs(pred-gt), dim=1)/dims['block']).item()
        
    return {'loss': loss,
            'gt': gt,
            'pred': pred}        

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