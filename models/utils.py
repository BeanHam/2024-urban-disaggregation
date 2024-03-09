import os
import torch
import numpy as np
import torch.nn.functional as F
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
# Prepare Data
# ---------------------
class taxi_data(torch.utils.data.Dataset):
    
    """
    Prepare taxi data
    """
    
    def __init__(self, 
                 puma, 
                 nta,
                 tract,
                 block,
                 extreme,
                 chunk_size):
        super(taxi_data, self).__init__()
        
        self.puma = puma
        self.nta = nta
        self.tract = tract
        self.block = block
        self.extreme = extreme
        self.chunk_size = chunk_size
        
    def __getitem__(self, index):
        
        ## batch attributes
        max_index = np.min([len(self.puma), index+self.chunk_size])
        batch_puma = self.puma[index:max_index]
        batch_nta = self.nta[index:max_index]
        batch_tract = self.tract[index:max_index]
        batch_block = self.block[index:max_index]
        batch_extreme = self.extreme[index:max_index]
        
        return batch_puma, \
               batch_nta, \
               batch_tract, \
               batch_block, \
               batch_extreme

    def __len__(self):
        return len(self.puma)

# ---------------------
# Load Data
# ---------------------
def load_data(parameters):
    
    """
    Function to load datasets
    
    Arg:
        - parameters: parameter json file
    """
    
    ## data path
    path = parameters['path']
    chunk_size = parameters['chunk_size']
    puma_path = path+'/attributes/puma'    
    nta_path = path+'/attributes/nta'
    tract_path = path+'/attributes/tract'
    block_path = path+'/attributes/block'
    extreme_path = path+'/attributes/extreme'
    
    ## linkage path
    puma_nta_path = path+'/linkages/puma_nta.npy'
    puma_tract_path = path+'/linkages/puma_tract.npy'
    puma_block_path = path+'/linkages/puma_block.npy'
    puma_extreme_path = path+'/linkages/puma_extreme.npy'
    nta_tract_path = path+'/linkages/nta_tract.npy'
    nta_block_path = path+'/linkages/nta_block.npy'
    nta_extreme_path = path+'/linkages/nta_extreme.npy'
    tract_block_path = path+'/linkages/tract_block.npy'
    tract_extreme_path = path+'/linkages/tract_extreme.npy'
    block_extreme_path = path+'/linkages/block_extreme.npy'
    
    ## load data
    X_puma_train = torch.from_numpy(np.load(puma_path+'_train.npy')).float()
    X_puma_val = torch.from_numpy(np.load(puma_path+'_val.npy')).float()
    X_puma_test = torch.from_numpy(np.load(puma_path+'_test.npy')).float()    
    X_nta_train = torch.from_numpy(np.load(nta_path+'_train.npy')).float()
    X_nta_val = torch.from_numpy(np.load(nta_path+'_val.npy')).float()
    X_nta_test = torch.from_numpy(np.load(nta_path+'_test.npy')).float()        
    X_tract_train = torch.from_numpy(np.load(tract_path+'_train.npy')).float()
    X_tract_val = torch.from_numpy(np.load(tract_path+'_val.npy')).float()
    X_tract_test = torch.from_numpy(np.load(tract_path+'_test.npy')).float()    
    X_block_train = torch.from_numpy(np.load(block_path+'_train.npy')).float()
    X_block_val = torch.from_numpy(np.load(block_path+'_val.npy')).float()
    X_block_test = torch.from_numpy(np.load(block_path+'_test.npy')).float()        
    X_extreme_train = torch.from_numpy(np.load(extreme_path+'_train.npy')).float()
    X_extreme_val = torch.from_numpy(np.load(extreme_path+'_val.npy')).float()
    X_extreme_test = torch.from_numpy(np.load(extreme_path+'_test.npy')).float()
    
    ## linkages
    puma_nta = torch.from_numpy(np.load(puma_nta_path)).float()
    puma_tract = torch.from_numpy(np.load(puma_tract_path)).float()
    puma_block = torch.from_numpy(np.load(puma_block_path)).float()
    puma_extreme = torch.from_numpy(np.load(puma_extreme_path)).float()
    nta_tract = torch.from_numpy(np.load(nta_tract_path)).float()
    nta_block = torch.from_numpy(np.load(nta_block_path)).float()
    nta_extreme = torch.from_numpy(np.load(nta_extreme_path)).float()
    tract_block = torch.from_numpy(np.load(tract_block_path)).float()
    tract_extreme = torch.from_numpy(np.load(tract_extreme_path)).float()
    block_extreme = torch.from_numpy(np.load(block_extreme_path)).float()
    
    linkages = {
        'puma_nta':puma_nta,
        'puma_tract':puma_tract,
        'puma_block':puma_block,
        'puma_extreme':puma_extreme,
        'nta_tract':nta_tract,
        'nta_block':nta_block,
        'nta_extreme':nta_extreme,
        'tract_block':tract_block,
        'tract_extreme':tract_extreme,
        'block_extreme':block_extreme
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
    dataset_train = taxi_data(X_puma_train, 
                              X_nta_train, 
                              X_tract_train, 
                              X_block_train, 
                              X_extreme_train,
                              chunk_size)
    dataset_val = taxi_data(X_puma_val, 
                            X_nta_val, 
                            X_tract_val, 
                            X_block_val, 
                            X_extreme_val,
                            chunk_size)
    dataset_test = taxi_data(X_puma_test, 
                             X_nta_test, 
                             X_tract_test, 
                             X_block_test, 
                             X_extreme_test,
                             chunk_size)
    
    return dataset_train, \
           dataset_val, \
           dataset_test, \
           X_low_max, \
           linkages


def cal_loss(
        preds, 
        puma, 
        nta,
        tract, 
        block, 
        extreme,
        parameters):
    
    block_dim = parameters['dims']['block']
    high = parameters['high']
    
    if high == 'nta': 
        loss = torch.mean(torch.sum(torch.abs(preds['nta']-nta), dim=-1)/block_dim)
    elif high == 'tract': 
        loss = torch.mean(torch.sum(torch.abs(preds['tract']-tract), dim=-1)/block_dim)
    elif high == 'block': 
        loss = torch.mean(torch.sum(torch.abs(preds['block']-block), dim=-1)/block_dim)
    else: 
        loss = torch.mean(torch.sum(torch.abs(preds['extreme']-extreme), dim=-1)/block_dim)
    
    return loss
    
def cal_loss_cot(
        preds, 
        puma, 
        nta,
        tract, 
        block, 
        extreme,
        parameters):
    
    block_dim = parameters['dims']['block']
    high = parameters['high']
        
    if high == 'nta':
        loss_puma = torch.mean(torch.sum(torch.abs(preds['puma']-puma), dim=-1)/block_dim)
        loss_nta = torch.mean(torch.sum(torch.abs(preds['nta']-nta), dim=-1)/block_dim)      
        loss = 10.0*loss_puma + 8.5*loss_nta    
    elif high == 'tract':
        loss_puma = torch.mean(torch.sum(torch.abs(preds['puma']-puma), dim=-1)/block_dim)
        loss_nta = torch.mean(torch.sum(torch.abs(preds['nta']-nta), dim=-1)/block_dim)
        loss_tract = torch.mean(torch.sum(torch.abs(preds['tract']-tract), dim=-1)/block_dim)      
        loss = 10.0*loss_puma + 8.5*loss_nta + 5.0*loss_tract
    elif high == 'block':
        loss_puma = torch.mean(torch.sum(torch.abs(preds['puma']-puma), dim=-1)/block_dim)
        loss_nta = torch.mean(torch.sum(torch.abs(preds['nta']-nta), dim=-1)/block_dim)
        loss_tract = torch.mean(torch.sum(torch.abs(preds['tract']-tract), dim=-1)/block_dim)
        loss_block = torch.mean(torch.sum(torch.abs(preds['block']-block), dim=-1)/block_dim)  
        loss = 10.0*loss_puma + 8.5*loss_nta + 5.0*loss_tract + 1.5*loss_block
    else:
        loss_puma = torch.mean(torch.sum(torch.abs(preds['puma']-puma), dim=-1)/block_dim)
        loss_nta = torch.mean(torch.sum(torch.abs(preds['nta']-nta), dim=-1)/block_dim)
        loss_tract = torch.mean(torch.sum(torch.abs(preds['tract']-tract), dim=-1)/block_dim)
        loss_block = torch.mean(torch.sum(torch.abs(preds['block']-block), dim=-1)/block_dim)  
        loss_extreme = torch.mean(torch.sum(torch.abs(preds['extreme']-extreme), dim=-1)/block_dim)         
        loss = 10.0*loss_puma + 8.5*loss_nta + 5.0*loss_tract + 1.5*loss_block + loss_extreme

    return loss

def cal_loss_cot_rec(
        results, 
        puma, 
        nta,
        tract, 
        block, 
        extreme,
        parameters):
    
    block_dim = parameters['dims']['block']
    high = parameters['high']
                
    if high == 'nta':   
        ## prediction loss        
        loss_puma = torch.mean(torch.sum(torch.abs(results['puma']-puma), dim=-1)/block_dim)
        loss_nta = torch.mean(torch.sum(torch.abs(results['nta']-nta), dim=-1)/block_dim)
        
        ## reconstruction loss        
        for rec_puma in results['rec_pumas']: loss_puma+=torch.mean(torch.sum(torch.abs(rec_puma-puma), dim=-1)/block_dim)                
        loss_puma = loss_puma/(len(results['rec_pumas'])+1)
        
        loss = 10.0*loss_puma + 8.5*loss_nta
        
    elif high == 'tract':
        ## prediction loss        
        loss_puma = torch.mean(torch.sum(torch.abs(results['puma']-puma), dim=-1)/block_dim)
        loss_nta = torch.mean(torch.sum(torch.abs(results['nta']-nta), dim=-1)/block_dim)
        loss_tract = torch.mean(torch.sum(torch.abs(results['tract']-tract), dim=-1)/block_dim)
        
        ## reconstruction loss        
        for rec_puma in results['rec_pumas']: loss_puma+=torch.mean(torch.sum(torch.abs(rec_puma-puma), dim=-1)/block_dim)                
        for rec_nta in results['rec_ntas']: loss_nta+=torch.mean(torch.sum(torch.abs(rec_nta-nta), dim=-1)/block_dim)    
        loss_puma = loss_puma/(len(results['rec_pumas'])+1)
        loss_nta = loss_nta/(len(results['rec_ntas'])+1)
        
        loss = 10.0*loss_puma + 8.5*loss_nta + 5.0*loss_tract
        
    elif high == 'block': 
        ## prediction loss        
        loss_puma = torch.mean(torch.sum(torch.abs(results['puma']-puma), dim=-1)/block_dim)
        loss_nta = torch.mean(torch.sum(torch.abs(results['nta']-nta), dim=-1)/block_dim)
        loss_tract = torch.mean(torch.sum(torch.abs(results['tract']-tract), dim=-1)/block_dim)
        loss_block = torch.mean(torch.sum(torch.abs(results['block']-block), dim=-1)/block_dim)
        
        ## reconstruction loss        
        for rec_puma in results['rec_pumas']: loss_puma+=torch.mean(torch.sum(torch.abs(rec_puma-puma), dim=-1)/block_dim)                
        for rec_nta in results['rec_ntas']: loss_nta+=torch.mean(torch.sum(torch.abs(rec_nta-nta), dim=-1)/block_dim)    
        for rec_tract in results['rec_tracts']: loss_tract+=torch.mean(torch.sum(torch.abs(rec_tract-tract), dim=-1)/block_dim)    
        loss_puma = loss_puma/(len(results['rec_pumas'])+1)
        loss_nta = loss_nta/(len(results['rec_ntas'])+1)
        loss_tract = loss_tract/(len(results['rec_tracts'])+1)
        
        loss = 10.0*loss_puma + 8.5*loss_nta + 5.0*loss_tract + 1.5*loss_block
        
    else:
        ## prediction loss
        loss_puma = torch.mean(torch.sum(torch.abs(results['puma']-puma), dim=-1)/block_dim)
        loss_nta = torch.mean(torch.sum(torch.abs(results['nta']-nta), dim=-1)/block_dim)
        loss_tract = torch.mean(torch.sum(torch.abs(results['tract']-tract), dim=-1)/block_dim)
        loss_block = torch.mean(torch.sum(torch.abs(results['block']-block), dim=-1)/block_dim)  
        loss_extreme = torch.mean(torch.sum(torch.abs(results['extreme']-extreme), dim=-1)/block_dim) 
        
        ## reconstruction loss
        for rec_puma in results['rec_pumas']: loss_puma+=torch.mean(torch.sum(torch.abs(rec_puma-puma), dim=-1)/block_dim)                
        for rec_nta in results['rec_ntas']: loss_nta+=torch.mean(torch.sum(torch.abs(rec_nta-nta), dim=-1)/block_dim)    
        for rec_tract in results['rec_tracts']: loss_tract+=torch.mean(torch.sum(torch.abs(rec_tract-tract), dim=-1)/block_dim)    
        for rec_block in results['rec_blocks']: loss_block+=torch.mean(torch.sum(torch.abs(rec_block-block), dim=-1)/block_dim)
    
        ## average
        loss_puma = loss_puma/(len(results['rec_pumas'])+1)
        loss_nta = loss_nta/(len(results['rec_ntas'])+1)
        loss_tract = loss_tract/(len(results['rec_tracts'])+1)
        loss_block = loss_block/(len(results['rec_blocks'])+1)
        
        loss = 10.0*loss_puma + 8.5*loss_nta + 5.0*loss_tract + 1.5*loss_block + loss_extreme

    return loss

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
    cot = parameters['cot']
    rec = parameters['rec']
    chunk_size = parameters['chunk_size']
    
    for i in range(0, len(dataset), batch_size):
        
        ## batch data
        indices = range(i, min(len(dataset)-chunk_size, i+batch_size))
        puma, nta, tract, block, extreme = zip(*[dataset[j] for j in indices])
        puma = torch.stack(puma).squeeze_(-1).to(device)
        nta = torch.stack(nta).squeeze_(-1).to(device)
        tract = torch.stack(tract).squeeze_(-1).to(device)
        block = torch.stack(block).squeeze_(-1).to(device)
        extreme = torch.stack(extreme).squeeze_(-1).to(device)
        
        ## prediction
        results = model(puma)
        
        ## loss calculation
        if (cot == 'no') and (rec == 'no'):
            loss=cal_loss(results, puma, nta,tract, block, extreme,parameters)        
        if (cot == 'yes') and (rec == 'no'):
            loss=cal_loss_cot(results, puma, nta,tract, block, extreme,parameters)
        if (cot == 'yes') and (rec != 'no'):
            loss=cal_loss_cot_rec(results, puma, nta,tract, block, extreme,parameters)
        
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
    preds = []
    high = parameters['high']
    block_dim = parameters['dims']['block']
    chunk_size = parameters['chunk_size']
    outer_indices = [k for k in range(0, len(dataset), chunk_size)]
    
    ## iterate through evaluation dataset
    for i in range(0, len(outer_indices), batch_size):

        ## batch data
        inner_indices = range(i, min(len(outer_indices), i+batch_size))
        puma, nta, tract, block, extreme = zip(*[dataset[outer_indices[j]] for j in inner_indices])
                
        ## padding
        puma = list(puma)
        nta = list(nta)
        tract = list(tract)
        block = list(block)
        extreme = list(extreme)
        for data in [puma, nta, tract, block, extreme]:
            for i in range(len(data)):
                if data[i].size(0) != chunk_size:
                    data[i] = F.pad(data[i],(0,0,0,0,0,chunk_size-data[i].size(0)))
           
        puma = torch.stack(puma).squeeze_(-1).to(device)
        nta = torch.stack(nta).squeeze_(-1).to(device)
        tract = torch.stack(tract).squeeze_(-1).to(device)
        block = torch.stack(block).squeeze_(-1).to(device)
        extreme = torch.stack(extreme).squeeze_(-1).to(device)
        
        ## prediction
        with torch.no_grad():
            results = model(puma)
            
        if high == 'nta': preds.append(results['nta'])
        elif high == 'tract': preds.append(results['tract'])
        elif high == 'block': preds.append(results['block'])
        else: preds.append(results['extreme'])            
                
    ## aggregate
    preds = torch.cat(preds).cpu()
    preds = preds.reshape(-1, preds.size(-1), 1)[:len(dataset)]
    if high == 'nta':
        gts = dataset.nta.cpu()     
        loss = torch.mean(torch.sum(torch.abs(preds-gts), dim=1)/block_dim).item()
    elif high == 'tract':
        gts = dataset.tract.cpu()
        loss = torch.mean(torch.sum(torch.abs(preds-gts), dim=1)/block_dim).item()
    elif high == 'block':
        gts = dataset.block.cpu()
        loss = torch.mean(torch.sum(torch.abs(preds-gts), dim=1)/block_dim).item()
    else:
        gts = dataset.extreme.cpu()
        loss = torch.mean(torch.sum(torch.abs(preds-gts), dim=1)/block_dim).item()
        
    return {'loss': loss,
            'gts': gts,
            'preds': preds}

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