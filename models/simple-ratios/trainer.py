import numpy as np
import pandas as pd
import torch
import json
from itertools import combinations
from utils import *
from tqdm import tqdm

def main():
           
    ################# parameters #################
    print('Load Parameters...')
    with open('config.json') as json_file:
        parameters = json.load(json_file)
    names = ['puma', 'nta', 'tract', 'block', 'extreme']
    training_portions = parameters['training_portions']
    all_losses = []
    criterion = torch.nn.L1Loss()
    
    print('Inferencing...')
    for low_res_name, super_res_name in combinations(names, 2):

        ## load data
        print('===================================')
        print(f'{low_res_name} -> {super_res_name}...')
        print('===================================')
        
        losses = []

        for training_portion in training_portions:
            parameters['training_portion'] = training_portion
            print(f'Training Portion: {training_portion}...')
            ## load data
            dataset_train, _, dataset_test = load_data(low_res_name, super_res_name, parameters)
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
            pred = []
            test_new = dataset_test.att_low.clone().unsqueeze_(-1)
            pred = torch.stack([torch.sum(test_new[i]*ratio_matrix, axis=0) for i in range(len(dataset_test))])
            true = dataset_test.att_super
            losses.append(criterion(pred, true).item()) 
        
        all_losses.append(losses)

    np.save('all_losses.npy', np.array(all_losses))

    ## save results
    losses = pd.DataFrame(np.array(all_losses),
                          columns=['propotion_1', 'propotion_0.5', 'propotion_0.1','propotion_0.05','propotion_0.01'],
                          index=['puma_nta', 'puma_tract', 'puma_block', 'puma_extreme', 
                                 'nta_tract', 'nta_block', 'nta_extreme', 
                                 'tract_block', 'tract_extreme',
                                 'block_extreme'])
    losses.to_csv('results.csv')
    print('Done...')
    
if __name__ == "__main__":
    main()