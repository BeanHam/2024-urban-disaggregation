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
                 source_extreme,
                 chunk_size):
        super(taxi_data, self).__init__()
        
        self.source_puma = source_puma
        self.source_nta = source_nta
        self.source_tract = source_tract
        self.source_block = source_block
        self.source_extreme = source_extreme        
        self.chunk_size = chunk_size
        
    def __getitem__(self, index):
        
        ## batch attributes
        max_index = np.min([len(self.source_puma), index+self.chunk_size])
        batch_source_puma = self.source_puma[index:max_index]
        batch_source_nta = self.source_nta[index:max_index]
        batch_source_tract = self.source_tract[index:max_index]
        batch_source_block = self.source_block[index:max_index]
        batch_source_extreme = self.source_extreme[index:max_index]        
        
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
    chunk_size = parameters['chunk_size']
    source_path = parameters['source_path']
    source_puma_path = source_path+'/attributes/puma'    
    source_nta_path = source_path+'/attributes/nta'
    source_tract_path = source_path+'/attributes/tract'
    source_block_path = source_path+'/attributes/block'
    source_extreme_path = source_path+'/attributes/extreme'
    
    puma_nta_linkage_path = source_path+'/linkages/puma_nta.npy'
    puma_tract_linkage_path = source_path+'/linkages/puma_tract.npy'
    puma_block_linkage_path = source_path+'/linkages/puma_block.npy'
    puma_extreme_linkage_path = source_path+'/linkages/puma_extreme.npy'
    nta_tract_linkage_path = source_path+'/linkages/nta_tract.npy'
    nta_block_linkage_path = source_path+'/linkages/nta_block.npy'
    nta_extreme_linkage_path = source_path+'/linkages/nta_extreme.npy'
    tract_block_linkage_path = source_path+'/linkages/tract_block.npy'
    tract_extreme_linkage_path = source_path+'/linkages/tract_extreme.npy'
    block_extreme_linkage_path = source_path+'/linkages/block_extreme.npy'
    
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
    
    ## linkages
    puma_nta_linkage = torch.from_numpy(np.load(puma_nta_linkage_path)).float()
    puma_tract_linkage = torch.from_numpy(np.load(puma_tract_linkage_path)).float()
    puma_block_linkage = torch.from_numpy(np.load(puma_block_linkage_path)).float()
    puma_extreme_linkage = torch.from_numpy(np.load(puma_extreme_linkage_path)).float()
    nta_tract_linkage = torch.from_numpy(np.load(nta_tract_linkage_path)).float()
    nta_block_linkage = torch.from_numpy(np.load(nta_block_linkage_path)).float()
    nta_extreme_linkage = torch.from_numpy(np.load(nta_extreme_linkage_path)).float()
    tract_block_linkage = torch.from_numpy(np.load(tract_block_linkage_path)).float()
    tract_extreme_linkage = torch.from_numpy(np.load(tract_extreme_linkage_path)).float()
    block_extreme_linkage = torch.from_numpy(np.load(block_extreme_linkage_path)).float()
    linkages = {
        'puma_nta_linkage':puma_nta_linkage,
        'puma_tract_linkage':puma_tract_linkage,
        'puma_block_linkage':puma_block_linkage,
        'puma_extreme_linkage':puma_extreme_linkage,
        'nta_tract_linkage':nta_tract_linkage,
        'nta_block_linkage':nta_block_linkage,
        'nta_extreme_linkage':nta_extreme_linkage,
        'tract_block_linkage':tract_block_linkage,
        'tract_extreme_linkage':tract_extreme_linkage,
        'block_extreme_linkage':block_extreme_linkage
    }
    
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
    dataset_train = taxi_data(X_puma_train, X_nta_train, X_tract_train, X_block_train, X_extreme_train, chunk_size)
    dataset_val = taxi_data(X_puma_val, X_nta_val, X_tract_val, X_block_val, X_extreme_val, chunk_size)
    dataset_test = taxi_data(X_puma_test, X_nta_test, X_tract_test, X_block_test, X_extreme_test, chunk_size)
    
    return dataset_train, dataset_val, dataset_test, X_low_max, linkages

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
    chunk_size = parameters['chunk_size']
    dims = parameters['dims']
    high = parameters['high']    
    for i in range(0, len(dataset)-chunk_size, batch_size):
        
        ## batch data
        indices = range(i, min(len(dataset)-chunk_size, i+batch_size))      
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
            rec_pumas = results['rec_pumas']       
            
            ## loss
            loss_puma = torch.mean(torch.sum(torch.abs(puma-source_puma),dim=-1)/dims['block'])
            loss_nta = torch.mean(torch.sum(torch.abs(nta-source_nta),dim=-1)/dims['block'])
            
            ## reconstruction loss
            for rec_puma in rec_pumas:
                loss_puma+=torch.mean(torch.sum(torch.abs(rec_puma-source_puma),dim=-1)/dims['block'])                                
            loss_puma = loss_puma/(len(rec_pumas)+1)                
            
            loss = 10.0*loss_puma + 8.5*loss_nta
            
        elif high == 'tract':
            puma = results['puma']
            nta = results['nta']
            tract = results['tract']
            rec_pumas = results['rec_pumas']
            rec_ntas = results['rec_ntas']
            
            ## loss
            loss_puma = torch.mean(torch.sum(torch.abs(puma-source_puma),dim=-1)/dims['block'])
            loss_nta = torch.mean(torch.sum(torch.abs(nta-source_nta),dim=-1)/dims['block'])
            loss_tract = torch.mean(torch.sum(torch.abs(tract-source_tract),dim=-1)/dims['block'])
            
            ## reconstruction loss
            for rec_puma in rec_pumas:
                loss_puma+=torch.mean(torch.sum(torch.abs(rec_puma-source_puma),dim=-1)/dims['block'])                                
            loss_puma = loss_puma/(len(rec_pumas)+1)
            
            for rec_nta in rec_ntas:
                loss_nta+=torch.mean(torch.sum(torch.abs(rec_nta-source_nta),dim=-1)/dims['block'])
            loss_nta = loss_nta/(len(rec_ntas)+1)
            
            loss = 10.0*loss_puma + 8.5*loss_nta + 5.0*loss_tract            
        
        elif high == 'block':
            puma = results['puma']
            nta = results['nta']
            tract = results['tract']
            block = results['block']
            rec_pumas = results['rec_pumas']
            rec_ntas = results['rec_ntas']
            rec_tracts = results['rec_tracts'] 
        
            ## loss
            loss_puma = torch.mean(torch.sum(torch.abs(puma-source_puma),dim=-1)/dims['block'])
            loss_nta = torch.mean(torch.sum(torch.abs(nta-source_nta),dim=-1)/dims['block'])
            loss_tract = torch.mean(torch.sum(torch.abs(tract-source_tract),dim=-1)/dims['block'])
            loss_block = torch.mean(torch.sum(torch.abs(block-source_block),dim=-1)/dims['block'])
            
            ## reconstruction loss
            for rec_puma in rec_pumas:
                loss_puma+=torch.mean(torch.sum(torch.abs(rec_puma-source_puma),dim=-1)/dims['block'])                                
            loss_puma = loss_puma/(len(rec_pumas)+1)
            
            for rec_nta in rec_ntas:
                loss_nta+=torch.mean(torch.sum(torch.abs(rec_nta-source_nta),dim=-1)/dims['block'])
            loss_nta = loss_nta/(len(rec_ntas)+1)
            
            for rec_tract in rec_tracts:
                loss_tract+=torch.mean(torch.sum(torch.abs(rec_tract-source_tract),dim=-1)/dims['block'])
            loss_tract = loss_tract/(len(rec_tracts)+1) 
            
            loss = 10.0*loss_puma + 8.5*loss_nta + 5.0*loss_tract + 1.5*loss_block        
        else:            
            puma = results['puma']
            nta = results['nta']
            tract = results['tract']
            block = results['block']
            extreme = results['extreme']
            rec_pumas = results['rec_pumas']
            rec_ntas = results['rec_ntas']  
            rec_tracts = results['rec_tracts']
            rec_blocks = results['rec_blocks']
            
            ## loss
            loss_puma = torch.mean(torch.sum(torch.abs(puma-source_puma),dim=-1)/dims['block'])
            loss_nta = torch.mean(torch.sum(torch.abs(nta-source_nta),dim=-1)/dims['block'])
            loss_tract = torch.mean(torch.sum(torch.abs(tract-source_tract),dim=-1)/dims['block'])
            loss_block = torch.mean(torch.sum(torch.abs(block-source_block),dim=-1)/dims['block'])
            loss_extreme = torch.mean(torch.sum(torch.abs(extreme-source_extreme),dim=-1)/dims['block'])
            
            ## reconstruction loss
            for rec_puma in rec_pumas:
                loss_puma+=torch.mean(torch.sum(torch.abs(rec_puma-source_puma),dim=-1)/dims['block'])                                
            loss_puma = loss_puma/(len(rec_pumas)+1)
            
            for rec_nta in rec_ntas:
                loss_nta+=torch.mean(torch.sum(torch.abs(rec_nta-source_nta),dim=-1)/dims['block'])
            loss_nta = loss_nta/(len(rec_ntas)+1)
            
            for rec_tract in rec_tracts:
                loss_tract+=torch.mean(torch.sum(torch.abs(rec_tract-source_tract),dim=-1)/dims['block'])
            loss_tract = loss_tract/(len(rec_tracts)+1)
            
            for rec_block in rec_blocks:
                loss_block+=torch.mean(torch.sum(torch.abs(rec_block-source_block),dim=-1)/dims['block'])
            loss_block = loss_block/(len(rec_blocks)+1)
                                 
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
    chunk_size = parameters['chunk_size']
    outer_indices = [k for k in range(0, len(dataset)-chunk_size+1, chunk_size)]
    
    ## iterate through evaluation dataset
    for i in range(0, len(outer_indices), batch_size):
        
        ## batch data
        inner_indices = range(i, min(len(outer_indices), i+batch_size))
        source_puma, source_nta, source_tract, source_block, source_extreme= zip(*[dataset[outer_indices[j]] for j in inner_indices])
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
        pred = pred.reshape(-1, pred.size(-1))
        gt = dataset.source_nta.squeeze_(-1)[:len(pred)]
        loss = torch.mean(torch.sum(torch.abs(pred-gt), dim=1)/dims['block']).item()
    elif high == 'tract':
        pred = torch.cat(pred).squeeze_(1).cpu()
        pred = pred.reshape(-1, pred.size(-1))
        gt = dataset.source_tract.squeeze_(-1)[:len(pred)]
        loss = torch.mean(torch.sum(torch.abs(pred-gt), dim=1)/dims['block']).item()
    elif high == 'block':
        pred = torch.cat(pred).squeeze_(1).cpu()
        pred = pred.reshape(-1, pred.size(-1))
        gt = dataset.source_block.squeeze_(-1)[:len(pred)]
        loss = torch.mean(torch.sum(torch.abs(pred-gt), dim=1)/dims['block']).item()
    else:
        pred = torch.cat(pred).squeeze_(1).cpu()
        pred = pred.reshape(-1, pred.size(-1))
        gt = dataset.source_extreme.squeeze_(-1)[:len(pred)]
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
