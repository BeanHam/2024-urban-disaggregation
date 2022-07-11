import numpy as np
import pandas as pd
import torch
import os
from itertools import combinations
from utils import *

def main():
           
    ################# parameters #################
    print('Load Parameters...')
    data_path = '../../data-processing/reality'
    names = ['puma', 'nta', 'tract', 'block']
    losses = []
    
    print('Inferencing...')
    for low_res_name, super_res_name in combinations(names, 2):
        
        ## load data
        att_low_res_path = data_path+'/attributes/'+low_res_name+'.npy'
        adj_low_res_path = data_path+'/adjacencies/'+low_res_name+'.npy'
        att_super_res_path = data_path+'/attributes/'+super_res_name+'.npy'
        adj_super_res_path = data_path+'/adjacencies/'+super_res_name+'.npy'
        linkage_path = data_path+'/linkages/'+low_res_name+'_'+super_res_name+'.npy'
        dataset_train, dataset_val, dataset_test = load_data(att_low_res_path, 
                                                             adj_low_res_path, 
                                                             att_super_res_path, 
                                                             adj_super_res_path,
                                                             linkage_path)
    
        ## load data
        low_res_data = np.load(f'{data_path}/attributes/{low_res_name}.npy')
        low_res_geo = pd.read_csv(f'{data_path}/geodata/{low_res_name}.csv')
        super_res_data = np.load(f'{data_path}/attributes/{super_res_name}.npy')
        super_res_geo = pd.read_csv(f'{data_path}/geodata/{super_res_name}.csv')
        linkage = np.load(f'{data_path}/linkages/{low_res_name}_{super_res_name}.npy')
        X_super_max = torch.max(torch.from_numpy(np.load(att_super_res_path)).float()).numpy()
    
        ## calculate ratio
        ratio_matrix = np.zeros_like(super_res_data)
        for i in range(low_res_data.shape[1]):
            low_res_sub = low_res_data[:,i].reshape(-1,1)
            linkage_sub = np.where(linkage[i]==1)[0]
            super_res_sub = super_res_data[:, linkage_sub]
            ratio = super_res_sub/low_res_sub
            ratio[np.isnan(ratio)] = 0
            ratio[np.isinf(ratio)] = 0
            ratio_matrix[:, linkage_sub] = ratio
            
        ## mean ratios
        mean_ratios = np.mean(ratio_matrix, axis=0)
        
        ## prediction
        criterion = nn.L1Loss()
        pred=((dataset_test.att_low*linkage)*mean_ratios.reshape(1,-1))
        pred = torch.sum(pred, axis=1, keepdim=True).transpose(-2,-1)
        loss = torch.mean(torch.abs(pred - dataset_test.att_super))
        losses.append(loss.item()*X_super_max) 
        
    ## save results
    losses = pd.DataFrame(np.array(losses).reshape(1,-1),
                          columns=['puma_nta', 'puma_tract', 'puma_block',
                                   'nta_tract', 'nta_block', 'tract_block'])
    losses.to_csv(f'results.csv')
    print('Done...')
    
if __name__ == "__main__":
    main()