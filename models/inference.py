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
    parser.add_argument('--rec', required=True, help='whether or not to reconstruct')
    parser.add_argument('--local', required=True, help='whether or not to use local attention')  
    args = parser.parse_args()
    data = args.data
    rec = args.rec
    local = args.local  

    # ----------------
    # Load Parameters
    # ----------------
    with open('config.json') as json_file:
        parameters = json.load(json_file)
    
    root = parameters['root']
    dims = parameters['dims']
    batch_size = parameters['batch_size']
    parameters['rec'] = rec
    parameters['local'] = local
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
    for coarse, fine in resolutions:
        
        # ----------------
        # Inference
        # ----------------    
        print('   =======================')
        print(f'{coarse} -> {fine}...')
        parameters['coarse'] = coarse
        parameters['fine'] = fine
        parameters['low'] = dims[coarse]
        parameters['high'] = dims[fine]
        parameters['path'] = root+data
        _, _, _, _, linkages = load_data(parameters) 
        losses = []
        
        # ----------------
        # Initialize Model
        # ----------------
        print('   ---Initialize Model...')
        if fine == 'nta':
            linkage = linkages['puma_nta'].to(device)
            model = puma_nta(linkage, rec, parameters).to(device)
        elif fine == 'tract':
            linkage = linkages['puma_tract'].to(device)
            model = puma_tract(linkage, rec, parameters).to(device)
        else:
            linkage = linkages['puma_block'].to(device)
            model = puma_block(linkage, rec, parameters).to(device)
        
        model.load_state_dict(torch.load(f'model_state/{data}_{coarse}_{fine}_rec_{rec}_local_{local}'))
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
    all_losses.to_csv(f'{data}_results_rec_{rec}_local_{local}.csv')
    print('Done...')
    
if __name__ == "__main__":
    main()
