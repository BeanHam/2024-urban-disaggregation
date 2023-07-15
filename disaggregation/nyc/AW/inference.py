import numpy as np
import pandas as pd
import argparse
import torch
import json
from utils import *

def main():           

    #-------------------------
    # arguments
    #-------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='filename of test data')
    args = parser.parse_args()
    data = args.data

    # ----------------
    # Load Parameters
    # ----------------
    print('Load Parameters...')
    with open('config.json') as json_file:
        parameters = json.load(json_file)
    root = parameters['root']
    factor = parameters['factor']
    resolutions = [('puma', 'nta'), 
                   ('puma', 'tract'),
                   ('puma', 'block'),
                   ('puma', 'extreme')]
    domains = ['taxi', 'bikeshare', '911-call']
    all_losses = []
    
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
        parameters['source_path'] = root+data
        geodata = pd.read_csv(root+data+f'/geodata/{high}.csv')
        _, _, _, linkage = load_data(low, high, parameters)        
        areas = geodata.shape_area.values.reshape(1,-1)
        ratio_matrix = (linkage*areas)/torch.sum(linkage*areas, dim=1,keepdim=True)
        
        # ----------------
        # target data prediction
        # ----------------
        for domain in domains:
            parameters['source_path'] = root+domain
            _, _, target_test,_ = load_data(low, high, parameters)
            test = target_test.source_low.clone().unsqueeze_(-1)
            pred = torch.stack([torch.sum(test[i]*ratio_matrix, axis=0) for i in range(len(test))])
            true = target_test.source_high
            losses.append(torch.mean(torch.sum(torch.abs(pred-true), dim=1)/factor).item())
            
        # ----------------
        # Save Results
        # ----------------
        all_losses.append(losses)

    ## save results
    all_losses = pd.DataFrame(
        np.array(all_losses), 
        index=resolutions,
        columns=domains
    )
    all_losses.to_csv(f'{data}_results.csv')
    print('Done...')
    
if __name__ == "__main__":
    main()