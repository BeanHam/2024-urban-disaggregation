import numpy as np
import pandas as pd
import torch
import json
import os
from itertools import combinations
from utils import *
from tqdm import tqdm

def main():
           
    ################# parameters #################
    print('Load Parameters...')
    with open('parameters.json') as json_file:
        parameters = json.load(json_file)
    names = ['puma', 'nta', 'tract', 'block']
    losses = []
    criterion = nn.L1Loss()
    
    print('Inferencing...')
    for low_res_name, super_res_name in combinations(names, 2):
        
        print(f'{low_res_name} -> {super_res_name}...')
        ## load data
        dataset_train, dataset_val, dataset_test = load_data(low_res_name, super_res_name, parameters)
        low_res_data = dataset_train.att_low
        super_res_data = dataset_train.att_super
        linkage = dataset_train.linkage

        ## ratio
        ratio_matrix = torch.zeros_like(linkage)
        for i in tqdm(range(low_res_data.size(0))):
            proj = (low_res_data[i].unsqueeze(-1)*linkage)
            rep_super = super_res_data[i].unsqueeze(0).repeat(low_res_data.size(1),1)
            ratio = rep_super/proj
            ratio[torch.isinf(ratio)] = 0
            ratio[torch.isnan(ratio)] = 0    
            ratio_matrix += ratio
        ratio_matrix /= low_res_data.size(0)
            
        ## prediction
        pred = torch.sum((dataset_test.att_low.unsqueeze(-1)*ratio_matrix), axis=1)
        true = dataset_test.att_super
        losses.append(criterion(pred, true).item()) 
        
    ## save results
    losses = pd.DataFrame(np.array(losses).reshape(1,-1),
                          columns=['puma_nta', 'puma_tract', 'puma_block',
                                   'nta_tract', 'nta_block', 'tract_block'])
    losses.to_csv(f'results.csv')
    print('Done...')
    
if __name__ == "__main__":
    main()