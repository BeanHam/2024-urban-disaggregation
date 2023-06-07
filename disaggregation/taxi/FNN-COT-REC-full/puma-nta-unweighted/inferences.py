import numpy as np
import pandas as pd
import json
import argparse
from tqdm import tqdm
from itertools import combinations
from utils import *
from models import *

import warnings
warnings.filterwarnings("ignore")

def main():
    
    # ----------------
    # Load Parameters
    # ----------------
    with open('config.json') as json_file:
        parameters = json.load(json_file)
    
    root = parameters['root']
    data_name = parameters['data_name']
    area = parameters['area']
    dims = parameters['dims']
    batch_size = parameters['batch_size']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolutions = [('puma', 'nta')]
    domains = ['taxi', 'bikeshare', '911-call']
    all_losses = []
    
    # ----------------
    # Inference
    # ----------------
    print('Inferencing...')
    for low, high in resolutions:
        
        print('   =======================')
        print(f'   {low} -> {high}...')
        losses = []
        parameters['source_path'] = root+data_name
        _, _, _, _, linkages = load_data(low, high, parameters)
        low_size = dims[low]
        high_size = dims[high]
        hidden_sizes = [dims['nta']]
        linkages = [linkage.to(device) for linkage in linkages]
        
        # -----------------
        # Load Source Model
        # -----------------
        model = DisAgg(low_size, high_size, hidden_sizes, linkages).to(device)
        criterion = nn.L1Loss().to(device)
        model.load_state_dict(torch.load(f'model_state/{low}_{high}'))
        model.eval()
        
        # ----------------
        # Iterate Domains
        # ----------------
        for domain in domains:
            
            # ----------------
            # Load Target Data
            # ----------------
            parameters['source_path'] = root+domain
            _,_,dataset_test, low_max,_ = load_data(low, high, parameters)
            
            # ----------------
            # Prediction
            # ----------------
            _,pred_high, gt_high = evaluation(model, 
                                              criterion, 
                                              device, 
                                              batch_size, 
                                              dataset_test)
            losses.append(torch.sum(torch.abs(pred_high-gt_high)).item()*low_max/area)
    
        # ----------------
        # save results
        # ----------------
        all_losses.append(losses)
    
    all_losses = pd.DataFrame(np.stack(all_losses).T, 
                              columns=resolutions,
                              index=domains)
    all_losses.to_csv('disaggregation.csv')
    print('Done...')
    
if __name__ == "__main__":
    main()