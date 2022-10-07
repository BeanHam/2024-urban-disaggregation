import numpy as np
import torch
from sklearn.model_selection import train_test_split

class taxi_data(torch.utils.data.Dataset):
    
    """
    Prepare taxi data
    """
    
    def __init__(self, 
                 att_low, 
                 att_super, 
                 adj_low,
                 adj_super,
                 linkage):
        super(taxi_data, self).__init__()
        self.att_low = att_low
        self.att_super = att_super
        self.adj_low = adj_low
        self.adj_super = adj_super
        self.linkage = linkage
        
    def __getitem__(self, index):
        
        ## batch attributes
        batch_att_low = self.att_low[index]
        batch_att_super = self.att_super[index]
        
        return batch_att_low, batch_att_super, self.adj_low, self.adj_super, self.linkage

    def __len__(self):
        return len(self.att_low)
    


def load_data(low_res_name, super_res_name, parameters):
    
    """
    Function to load datasets
    
    Arg:
        - parameters: parameter json file
    """
    
    ## data path
    data_path = parameters['data_path']
    training_portion = parameters['training_portion']
    att_low_res_path = data_path+'attributes/'+low_res_name+'.npy'
    adj_low_res_path = data_path+'adjacencies/'+low_res_name+'.npy'
    att_super_res_path = data_path+'attributes/'+super_res_name+'.npy'
    adj_super_res_path = data_path+'adjacencies/'+super_res_name+'.npy'
    linkage_path = data_path+'linkages/'+low_res_name+'_'+super_res_name+'.npy'
    
    ## load data
    X_low = torch.from_numpy(np.load(att_low_res_path)).float()
    X_super = torch.from_numpy(np.load(att_super_res_path)).float()
    A_low = torch.from_numpy(np.load(adj_low_res_path)).float()
    A_super = torch.from_numpy(np.load(adj_super_res_path)).float()
    linkage = torch.from_numpy(np.load(linkage_path)).float()
    
    ## split train, val & test
    X_low_train, X_low_test, X_super_train, X_super_test = train_test_split(X_low, 
                                                                            X_super, 
                                                                            test_size=0.2, 
                                                                            random_state=1)
    X_low_train, X_low_val, X_super_train, X_super_val = train_test_split(X_low_train, 
                                                                          X_super_train, 
                                                                          test_size=0.1, 
                                                                          random_state=1)
    ## training portion
    indices = int(len(X_low_train)*training_portion)
    X_low_train = X_low_train[:indices]
    X_super_train = X_super_train[:indices]
    
    ## prepare data
    dataset_train = taxi_data(X_low_train, X_super_train, A_low, A_super, linkage)
    dataset_val = taxi_data(X_low_val, X_super_val, A_low, A_super, linkage)
    dataset_test = taxi_data(X_low_test, X_super_test, A_low, A_super, linkage)
    
    return dataset_train, dataset_val, dataset_test