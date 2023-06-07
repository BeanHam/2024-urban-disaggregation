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
    area = parameters['area']   
    batch_size = parameters['batch_size']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolutions = [('puma', 'nta'), 
                   ('puma', 'tract'),
                   ('puma', 'block'),
                   ('puma', 'extreme')]
    domains = ['bikeshare', '911-call']
    all_losses = []
    all_transformation_losses = []
    
    # ----------------
    # Inference
    # ----------------
    print('Inferencing...')
    for low, high in resolutions:
            
        print('   =======================')
        print(f'   {low} -> {high}...')
        losses = []
        transformation_losses = []
                   
        # -----------------
        # Load Source Model
        # -----------------
        model = DisAgg().to(device)
        criterion = nn.L1Loss().to(device)
        model.load_state_dict(torch.load(f'model_state/{low}_{high}'))
        model.eval()
        
        # ----------------
        # Iterate Domains
        # ----------------
        for domain in domains:
            
            # ----------------
            # Load Parameters
            # ----------------
            source_path = root+domain
            parameters['source_path'] = source_path
            boundaries = np.load(source_path+f'/img-data/{high}_boundaries.npy')
            test = np.load(source_path+f'/attributes/{high}_test.npy')
            
            # ----------------
            # Load Target Data
            # ----------------
            _,_,dataset_test, low_max = load_data(low, high, parameters)
            
            # ----------------
            # Prediction
            # ----------------
            _, pred_high, gt_high = evaluation(model,
                                               criterion, 
                                               device, 
                                               batch_size, 
                                               dataset_test)
            
            # ----------------
            # Pixels to Counts
            # ----------------
            pred_all, gt_all = count_error(pred_high, 
                                           gt_high,
                                           boundaries)
            losses.append(np.sum(np.abs(pred_all*low_max-test))/pred_all.shape[0]/3751)
            transformation_losses.append(np.sum(np.abs(gt_all*low_max-test))/area)
            
        # ----------------
        # save results
        # ----------------
        all_losses.append(losses)
        all_transformation_losses.append(transformation_losses)
    
    all_losses = pd.DataFrame(np.stack(all_losses).T, 
                              columns=resolutions,
                              index=domains)
    all_losses.to_csv('disaggregagtion.csv')
    np.save('transformation-losses.npy', all_transformation_losses)
    print('Done...')
    
if __name__ == "__main__":
    main()