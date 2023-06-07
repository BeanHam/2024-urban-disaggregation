import numpy as np
import pandas as pd
import torch
import json
from itertools import combinations
from utils import *
from tqdm import tqdm

def main():
           
    # ----------------
    # Load Parameters
    # ----------------
    print('Load Parameters...')
    with open('config.json') as json_file:
        parameters = json.load(json_file)
    root = parameters['root']
    data_name = parameters['data_name']
    area = parameters['area']
    resolutions = [('puma', 'nta'), 
                   ('puma', 'tract'),
                   ('puma', 'block'),
                   ('puma', 'extreme')]
    domains = ['taxi', 
               'bikeshare', 
               '911-call']
    all_losses = []
    criterion = torch.nn.L1Loss()
    
    # ----------------
    # Inference
    # ----------------
    print('Inferencing...')
    for low, high in resolutions:
    
        ## load data
        print('   =======================')
        print(f'   {low} -> {high}...')
        losses = []
        
        # ----------------
        # load source data
        # ----------------
        parameters['source_path'] = root+data_name
        _, _, _, linkage = load_data(low, high, parameters)
        ratio_matrix = linkage/torch.sum(linkage, dim=1, keepdim=True)
        
        # ----------------
        # target data prediction
        # ----------------
        for domain in domains:
            parameters['source_path'] = root+domain
            _, _, target_test,_ = load_data(low, high, parameters)
            test = target_test.source_low.clone().unsqueeze_(-1)
            pred = torch.stack([torch.sum(test[i]*ratio_matrix, axis=0) for i in range(len(test))])
            true = target_test.source_high
            losses.append(torch.sum(torch.abs(pred-true)).item()/pred.size(0)/3751)
            
        # ----------------
        # Save Results
        # ----------------
        all_losses.append(losses)

    ## save results
    all_losses = pd.DataFrame(np.array(all_losses).T, 
                              columns=resolutions,
                              index=domains)
    all_losses.to_csv('disaggregation.csv')
    print('Done...')
    
if __name__ == "__main__":
    main()