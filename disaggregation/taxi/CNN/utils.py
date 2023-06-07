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
                 source_high):
        super(taxi_data, self).__init__()
        
        self.source_low = source_low
        self.source_high = source_high
        
    def __getitem__(self, index):
        
        ## batch attributes
        batch_source_low = self.source_low[index]
        batch_source_high = self.source_high[index]
        
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
    source_path = parameters['source_path']
    source_low_path = source_path+'/img-data/'+low
    source_high_path = source_path+'/img-data/'+high
    
    ## load data
    source_low_train = torch.from_numpy(np.load(source_low_path+'_imgs_train.npy')).float()
    source_low_val = torch.from_numpy(np.load(source_low_path+'_imgs_val.npy')).float()
    source_low_test = torch.from_numpy(np.load(source_low_path+'_imgs_test.npy')).float()
    source_high_train = torch.from_numpy(np.load(source_high_path+'_imgs_train.npy')).float()
    source_high_val = torch.from_numpy(np.load(source_high_path+'_imgs_val.npy')).float()
    source_high_test = torch.from_numpy(np.load(source_high_path+'_imgs_test.npy')).float()
    
    source_low_max = np.max(
        [
            torch.max(source_low_train).item(),
            torch.max(source_low_val).item(),
            torch.max(source_low_test).item(),]
    )
    
    ## normalize
    source_low_train = source_low_train/source_low_max
    source_low_val = source_low_val/source_low_max
    source_low_test = source_low_test/source_low_max
    source_high_train = source_high_train/source_low_max
    source_high_val = source_high_val/source_low_max
    source_high_test = source_high_test/source_low_max
    
    ## prepare data
    dataset_train = taxi_data(source_low_train, source_high_train)
    dataset_val = taxi_data(source_low_val, source_high_val)
    dataset_test = taxi_data(source_low_test, source_high_test)
    
    return dataset_train, dataset_val, dataset_test, source_low_max


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
        source_low, source_high = zip(*[dataset[i] for i in indices])
        source_low = torch.stack(source_low).unsqueeze_(1).to(device)
        source_high = torch.stack(source_high).unsqueeze_(1).to(device)
        
        ## prediction
        pred = model(source_low)
        loss = criterion(source_high, pred)
        
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
    pred_high = []
    gt_high = []
    
    ## iterate through evaluation dataset
    for i in range(0, len(dataset), batch_size):
        
        ## batch data
        indices = range(i, min(len(dataset), i+batch_size))        
        source_low, source_high = zip(*[dataset[i] for i in indices])
        source_low = torch.stack(source_low).unsqueeze_(1).to(device)
        source_high = torch.stack(source_high).unsqueeze_(1).to(device)
        
        ## prediction
        with torch.no_grad():
            pred = model(source_low)
        pred_high.append(pred)
        gt_high.append(source_high)
    
    ## aggregate
    pred_high = torch.cat(pred_high).squeeze_(1)
    gt_high = torch.cat(gt_high).squeeze_(1)
    loss = criterion(pred_high, gt_high).cpu().item()
        
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
            
# ---------------------
# Error Calculation
# ---------------------
def count_error(pred_high,
                gt_high,
                boundaries):
    
        
    """
    Function to evaluate the model
    
    Arg:
        - model
        - criterion: loss function
        - batch_size
        - dataset: validation/test dataset
        
    """
    
    ## storage
    pred_all = []
    gt_all = []
    
    ## iterate each image
    for i in tqdm(range(len(pred_high))):
        
        pred_single = pred_high[i]
        gt_single = gt_high[i]
        
        pred_counts = []
        gt_counts = []
        
        ## iterate each boundary
        for bound in boundaries:
            pred_counts.append(torch.sum(pred_single[np.where(bound<255.)]).item())
            gt_counts.append(torch.sum(gt_single[np.where(bound<255.)]).item())
        pred_all.append(pred_counts)
        gt_all.append(gt_counts)
    
    ## stack & calculate error
    pred_all = np.stack(pred_all)
    gt_all = np.stack(gt_all)
    
    return pred_all, gt_all