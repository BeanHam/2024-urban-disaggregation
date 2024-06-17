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
                 time_steps):
        super(taxi_data, self).__init__()
        
        self.puma = puma
        self.nta = nta
        self.tract = tract
        self.block = block
        self.time_steps = time_steps
        
    def __getitem__(self, index):
        
        ## batch attributes
        max_index = np.min([len(self.puma), index+self.time_steps])
        batch_puma = self.puma[index:max_index]
        batch_nta = self.nta[index:max_index]
        batch_tract = self.tract[index:max_index]
        batch_block = self.block[index:max_index]
        
        return batch_puma, \
               batch_nta, \
               batch_tract, \
               batch_block

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
    time_steps = parameters['time_steps']
    puma_path = path+'/attributes/puma'    
    nta_path = path+'/attributes/nta'
    tract_path = path+'/attributes/tract'
    block_path = path+'/attributes/block'
    
    ## linkage path
    puma_nta_path = path+'/linkages/puma_nta.npy'
    puma_tract_path = path+'/linkages/puma_tract.npy'
    puma_block_path = path+'/linkages/puma_block.npy'
    nta_tract_path = path+'/linkages/nta_tract.npy'
    nta_block_path = path+'/linkages/nta_block.npy'
    tract_block_path = path+'/linkages/tract_block.npy'
    
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
    
    ## linkages
    puma_nta = torch.from_numpy(np.load(puma_nta_path)).float()
    puma_tract = torch.from_numpy(np.load(puma_tract_path)).float()
    puma_block = torch.from_numpy(np.load(puma_block_path)).float()
    nta_tract = torch.from_numpy(np.load(nta_tract_path)).float()
    nta_block = torch.from_numpy(np.load(nta_block_path)).float()
    tract_block = torch.from_numpy(np.load(tract_block_path)).float()
    
    linkages = {
        'puma_nta':puma_nta,
        'puma_tract':puma_tract,
        'puma_block':puma_block,
        'nta_tract':nta_tract,
        'nta_block':nta_block,
        'tract_block':tract_block
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
    
    ## prepare data
    dataset_train = taxi_data(X_puma_train, 
                              X_nta_train, 
                              X_tract_train, 
                              X_block_train, 
                              time_steps)
    dataset_val = taxi_data(X_puma_val, 
                            X_nta_val, 
                            X_tract_val, 
                            X_block_val, 
                            time_steps)
    dataset_test = taxi_data(X_puma_test, 
                             X_nta_test, 
                             X_tract_test, 
                             X_block_test, 
                             time_steps)
    
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
        parameters):
    
    block_dim = parameters['dims']['block']
    fine = parameters['fine']
    
    if fine == 'nta': 
        loss = torch.mean(torch.sum(torch.abs(preds['nta']-nta), dim=-1)/block_dim)
    elif fine == 'tract': 
        loss = torch.mean(torch.sum(torch.abs(preds['tract']-tract), dim=-1)/block_dim)
    elif fine == 'block': 
        loss = torch.mean(torch.sum(torch.abs(preds['block']-block), dim=-1)/block_dim)
    
    return loss    

def cal_loss_rec(results, 
                 puma, 
                 nta,
                 tract, 
                 block, 
                 parameters):
    
    puma_dim = parameters['dims']['puma']
    nta_dim = parameters['dims']['nta']
    tract_dim = parameters['dims']['tract']
    block_dim = parameters['dims']['block']
    fine = parameters['fine']
    
    if fine == 'nta':
        loss_nta = torch.mean(torch.sum(torch.abs(results['nta']-nta), dim=-1)/block_dim)
        loss_puma = torch.mean(torch.sum(torch.abs(results['rec_puma']-puma), dim=-1)/block_dim)
        loss = loss_puma + loss_nta
    elif fine == 'tract':
        loss_tract = torch.mean(torch.sum(torch.abs(results['tract']-tract), dim=-1)/block_dim)
        loss_puma = torch.mean(torch.sum(torch.abs(results['rec_puma']-puma), dim=-1)/block_dim)
        loss = 5*loss_puma + loss_tract
    elif fine == 'block':
        loss_block = torch.mean(torch.sum(torch.abs(results['block']-block), dim=-1)/block_dim)
        loss_puma = torch.mean(torch.sum(torch.abs(results['rec_puma']-puma), dim=-1)/block_dim)
        loss = 10*loss_puma + loss_block
        
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
    rec = parameters['rec']
    time_steps = parameters['time_steps']
    
    for i in range(0, len(dataset)-time_steps, batch_size):
        
        ## batch data
        indices = range(i, min(len(dataset)-time_steps, i+batch_size))
        puma, nta, tract, block = zip(*[dataset[j] for j in indices])
        puma = torch.stack(puma).squeeze_(-1).to(device)
        nta = torch.stack(nta).squeeze_(-1).to(device)
        tract = torch.stack(tract).squeeze_(-1).to(device)
        block = torch.stack(block).squeeze_(-1).to(device)
        
        ## prediction
        results = model(puma)
        
        ## loss calculation
        if rec == 'no':
            loss=cal_loss(results, puma, nta,tract, block,parameters)
        else: 
            loss=cal_loss_rec(results, puma, nta,tract, block,parameters)
        
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
    fine = parameters['fine']
    block_dim = parameters['dims']['block']
    time_steps = parameters['time_steps']
    outer_indices = [k for k in range(0, len(dataset), time_steps)]
    
    ## iterate through evaluation dataset
    for i in range(0, len(outer_indices), batch_size):

        ## batch data
        inner_indices = range(i, min(len(outer_indices), i+batch_size))
        puma, nta, tract, block = zip(*[dataset[outer_indices[j]] for j in inner_indices])
                
        ## padding
        puma = list(puma)
        nta = list(nta)
        tract = list(tract)
        block = list(block)
        for data in [puma, nta, tract, block]:
            for i in range(len(data)):
                if data[i].size(0) != time_steps:
                    data[i] = F.pad(data[i],(0,0,0,0,0,time_steps-data[i].size(0)))
           
        puma = torch.stack(puma).squeeze_(-1).to(device)
        nta = torch.stack(nta).squeeze_(-1).to(device)
        tract = torch.stack(tract).squeeze_(-1).to(device)
        block = torch.stack(block).squeeze_(-1).to(device)
        
        ## prediction
        with torch.no_grad():
            results = model(puma)
            
        if fine == 'nta': preds.append(results['nta'])
        elif fine == 'tract': preds.append(results['tract'])
        elif fine == 'block': preds.append(results['block'])
                
    ## aggregate
    preds = torch.cat(preds).cpu()
    preds = preds.reshape(-1, preds.size(-1), 1)[:len(dataset)]    
    if fine == 'nta':
        gts = dataset.nta.cpu()
        loss = torch.mean(torch.sum(torch.abs(preds-gts), dim=1)/block_dim).item()
    elif fine == 'tract':
        gts = dataset.tract.cpu()
        loss = torch.mean(torch.sum(torch.abs(preds-gts), dim=1)/block_dim).item()
    elif fine == 'block':
        gts = dataset.block.cpu()
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
