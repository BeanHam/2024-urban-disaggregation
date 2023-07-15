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
    resolutions = [('com', 'tract'),
                   ('com', 'block'),
                   ('com', 'extreme')]
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
        source_train, _, target_test, linkage = load_data(low, high, parameters)
        low_data = source_train.source_low
        high_data = source_train.source_high
        
        ## ratio
        ratio_matrix = torch.zeros_like(linkage)
        for i in range(low_data.size(0)):
            proj = (low_data[i].unsqueeze(-1)*linkage)
            rep_high = high_data[i].unsqueeze(0).repeat(low_data.size(1),1)
            ratio = rep_high/proj
            ratio[torch.isinf(ratio)] = 0
            ratio[torch.isnan(ratio)] = 0    
            ratio_matrix += ratio
        ratio_matrix /= low_data.size(0)
        
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
        index=resolutions
    )
    all_losses.to_csv(f'{data}_results.csv')
    print('Done...')
    
if __name__ == "__main__":
    main()