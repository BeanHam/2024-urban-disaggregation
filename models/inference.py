import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from utils import *
from models import *
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"

# ---------------------------
# seeding for reproducibility
# ---------------------------
seed = 100
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True)

def main():
    
    #-------------------------
    # arguments
    #-------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='data name')
    parser.add_argument('--nheads', required=True, help='data name')
    parser.add_argument('--cot', required=True, help='whether to do chain-of-training. Values: yes/no')
    parser.add_argument('--rec', required=True, help='which reconstruction to do. Values: no/bottomup/bridge/full')
    args = parser.parse_args()
    data = args.data
    nheads = int(args.nheads)
    cot = args.cot
    rec = args.rec

    # ----------------
    # Load Parameters
    # ----------------
    with open('config.json') as json_file:
        parameters = json.load(json_file)
    
    root = parameters['root']
    dims = parameters['dims']
    batch_size = parameters['batch_size']
    parameters['cot'] = cot
    parameters['rec'] = rec    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolutions = [('puma', 'nta'),
                   ('puma', 'tract'),
                   ('puma', 'block'),]
    domains = ['taxi', 
               'bikeshare', 
               '911']
    all_losses = []
    
    # -------------------
    # Iterate Resolutions
    # -------------------
    print('Inferencing...')    
    for low, high in resolutions:
        
        # ----------------
        # Inference
        # ----------------    
        print('   =======================')
        print(f'   {low} -> {high}...')
        parameters['low'] = low
        parameters['high'] = high
        parameters['path'] = root+data
        _, _, _, _, linkages = load_data(parameters) 
        losses = []
        
        # ----------------
        # Initialize Model
        # ----------------
        print('   ---Initialize Model...')
        if high == 'nta':
            linkages = [             
                linkages['puma_nta'].to(device)
            ]
            hidden_dims = [
                dims['puma'], 
                dims['nta']
            ]
            model = puma_nta(
                hidden_dims, 
                nheads,
                linkages, 
                rec).to(device)
        
        elif high == 'tract':
            linkages = [           
                linkages['puma_nta'].to(device), 
                linkages['puma_tract'].to(device), 
                linkages['nta_tract'].to(device)
            ]            
            hidden_dims = [
                dims['puma'], 
                dims['nta'], 
                dims['tract']
            ]
            model = puma_tract(
                hidden_dims, 
                nheads,
                linkages, 
                rec).to(device)
            
        else:
            linkages = [
                linkages['puma_nta'].to(device), 
                linkages['puma_tract'].to(device), 
                linkages['puma_block'].to(device),                        
                linkages['nta_tract'].to(device), 
                linkages['nta_block'].to(device), 
                linkages['tract_block'].to(device)
            ]            
            hidden_dims = [
                dims['puma'], 
                dims['nta'],
                dims['tract'], 
                dims['block']
            ]
            model = puma_block(
                hidden_dims, 
                nheads,
                linkages, 
                rec).to(device)
        
        model.load_state_dict(torch.load(f'model_state/{data}_{low}_{high}_cot_{cot}_rec_{rec}_nheads_{nheads}'))
        model.eval()
        
        # ----------------
        # target data prediction
        # ----------------
        for domain in domains:
            
            # ----------------
            # Load Target Data
            # ----------------
            parameters['path'] = root+domain
            _,_,dataset_test, low_max,_ = load_data(parameters)
            
            # ----------------
            # Prediction
            # ----------------            
            eval_results = evaluation(model, 
                                      device, 
                                      batch_size, 
                                      dataset_test,
                                      parameters)
            preds = eval_results['preds']
            gts = eval_results['gts']        
            losses.append(
                torch.mean(
                    torch.sum(torch.abs(preds-gts), dim=1)/dims['block']
                    ).item()*low_max
                    )
            
        all_losses.append(losses)
        
    # ----------------
    # save results
    # ----------------
    all_losses = pd.DataFrame(
        np.array(all_losses), 
        index=resolutions,
        columns=domains
    )
    all_losses.to_csv(f'{data}_results_nheads_{nheads}.csv')
    print('Done...')
    
if __name__ == "__main__":
    main()
