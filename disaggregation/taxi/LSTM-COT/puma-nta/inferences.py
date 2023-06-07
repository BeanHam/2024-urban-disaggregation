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
    chunk_size = parameters['chunk_size']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    low, high = 'puma', 'nta'
    domains = ['taxi', 'bikeshare', '911-call']
    
    # ----------------
    # Inference
    # ----------------
    print('Inferencing...')        
    print('   =======================')
    print(f'   {low} -> {high}...')
    losses = []
    low_size = dims[low]
    high_size = dims[high]
    hidden_sizes = [int(0.5*(low_size+high_size))] 
    
    # -----------------
    # Load Source Model
    # -----------------
    model = DisAgg(low_size, high_size, hidden_sizes).to(device)
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
        _,_,dataset_test, low_max = load_data(low, high, parameters)
        
        # ----------------
        # Prediction
        # ----------------
        eval_results = evaluation(model, 
                                  criterion, 
                                  device, 
                                  batch_size, 
                                  dataset_test,
                                  parameters)
        pred_high = eval_results['pred_nta']
        gt_high = eval_results['gt_nta']
        losses.append(torch.sum(torch.abs(pred_high-gt_high)).item()*low_max/pred_high.size(0)/3751)
    
    # ----------------
    # save results
    # ----------------
    losses = pd.DataFrame(np.array(losses).T, 
                          columns=[f'({low}, {high})'],
                          index=domains)
    losses.to_csv('disaggregation.csv')
    print('Done...')
    
if __name__ == "__main__":
    main()