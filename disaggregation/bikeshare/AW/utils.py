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
    source_low_path = source_path+'/attributes/'+low
    source_high_path = source_path+'/attributes/'+high
    linkage_path = source_path+'/linkages/'+low+'_'+high+'.npy'
    
    ## load data
    X_low_train = torch.from_numpy(np.load(source_low_path+'_train.npy')).float()
    X_low_val = torch.from_numpy(np.load(source_low_path+'_val.npy')).float()
    X_low_test = torch.from_numpy(np.load(source_low_path+'_test.npy')).float()
    X_high_train = torch.from_numpy(np.load(source_high_path+'_train.npy')).float()
    X_high_val = torch.from_numpy(np.load(source_high_path+'_val.npy')).float()
    X_high_test = torch.from_numpy(np.load(source_high_path+'_test.npy')).float()
    linkage = torch.from_numpy(np.load(linkage_path)).float()
    
    ## prepare data
    dataset_train = taxi_data(X_low_train, X_high_train)
    dataset_val = taxi_data(X_low_val, X_high_val)
    dataset_test = taxi_data(X_low_test, X_high_test)
    
    return dataset_train, dataset_val, dataset_test, linkage