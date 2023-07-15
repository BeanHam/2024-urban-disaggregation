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
    
    ## load data
    X_low_train = torch.from_numpy(np.load(source_low_path+'_train.npy')).float()
    X_low_val = torch.from_numpy(np.load(source_low_path+'_val.npy')).float()
    X_low_test = torch.from_numpy(np.load(source_low_path+'_test.npy')).float()
    X_high_train = torch.from_numpy(np.load(source_high_path+'_train.npy')).float()
    X_high_val = torch.from_numpy(np.load(source_high_path+'_val.npy')).float()
    X_high_test = torch.from_numpy(np.load(source_high_path+'_test.npy')).float()
    
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
    chunk_size = parameters['chunk_size']
    dims = parameters['dims']
    for i in range(0, len(dataset)-chunk_size, batch_size):
        
        ## batch data
        indices = range(i, min(len(dataset)-chunk_size, i+batch_size))      
        source_low, source_high = zip(*[dataset[j] for j in indices])
        source_low = torch.stack(source_low).squeeze_(-1).to(device)
        source_high = torch.stack(source_high).squeeze_(-1).to(device)
        
        ## prediction
        pred = model(source_low)
        loss = torch.mean(torch.sum(torch.abs(pred-source_high),dim=-1)/dims['block'])
        
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
    pred_high = []
    gt_high = []
    chunk_size = parameters['chunk_size']
    dims = parameters['dims']
    outer_indices = [k for k in range(0, len(dataset)-chunk_size+1, chunk_size)]
    
    ## iterate through evaluation dataset
    for i in range(0, len(outer_indices), batch_size):
        
        ## batch data
        inner_indices = range(i, min(len(outer_indices), i+batch_size))
        source_low, source_high = zip(*[dataset[outer_indices[j]] for j in inner_indices])
        source_low = torch.stack(source_low).squeeze_(-1).to(device)
        source_high = torch.stack(source_high).squeeze_(-1).to(device)
        
        ## prediction
        with torch.no_grad():
            pred = model(source_low)
        pred_high.append(pred)
        gt_high.append(source_high)
        
    ## aggregate
    pred_high = torch.cat(pred_high).squeeze_(1)
    gt_high = torch.cat(gt_high).squeeze_(1)
    
    ## calculate loss
    loss = torch.mean(torch.sum(torch.abs(pred_high-gt_high),dim=-1)/dims['block']).item()
    
    return loss, pred_high, gt_high

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
