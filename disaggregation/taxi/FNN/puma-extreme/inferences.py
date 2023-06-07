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
    low, high = 'puma', 'extreme'
    domains = ['taxi', 'bikeshare', '911-call']
    all_losses = []
    
    # ----------------
    # Inference
    # ----------------
    print('Inferencing...')        
    print('   =======================')
    print(f'   {low} -> {high}...')
    losses = []
    stds = []
    low_size = dims[low]
    high_size = dims[high]
    hidden_sizes = [dims['nta'], dims['tract'], dims['block']]        
    
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
        _,pred_extreme, gt_extreme = evaluation(model, 
                                                criterion, 
                                                device, 
                                                batch_size, 
                                                dataset_test)
        
        ## linkages
        source_path = parameters['source_path']
        nta_data_path = source_path+'/attributes/nta_test.npy'
        tract_data_path = source_path+'/attributes/tract_test.npy'
        block_data_path = source_path+'/attributes/block_test.npy'        
        nta_extreme_linkage_path = source_path+'/linkages/nta_extreme.npy'
        tract_extreme_linkage_path = source_path+'/linkages/tract_extreme.npy'
        block_extreme_linkage_path = source_path+'/linkages/block_extreme.npy'
        
        gt_nta = torch.from_numpy(np.load(nta_data_path)).float()/low_max
        gt_tract = torch.from_numpy(np.load(tract_data_path)).float()/low_max
        gt_block = torch.from_numpy(np.load(block_data_path)).float()/low_max
        gt_extreme = dataset_test.source_high.squeeze_(-1)
        nta_extreme_linkage = torch.from_numpy(np.load(nta_extreme_linkage_path)).float()
        tract_extreme_linkage = torch.from_numpy(np.load(tract_extreme_linkage_path)).float()
        block_extreme_linkage = torch.from_numpy(np.load(block_extreme_linkage_path)).float()        
        
        pred_extreme = pred_extreme.detach().cpu()
        pred_nta = pred_extreme@nta_extreme_linkage.T
        pred_tract = pred_extreme@tract_extreme_linkage.T
        pred_block = pred_extreme@block_extreme_linkage.T
        
        loss_nta = torch.sum(torch.abs(pred_nta-gt_nta)).item()*low_max/gt_nta.size(0)/3751
        loss_tract = torch.sum(torch.abs(pred_tract-gt_tract)).item()*low_max/gt_nta.size(0)/3751
        loss_block = torch.sum(torch.abs(pred_block-gt_block)).item()*low_max/gt_nta.size(0)/3751
        loss_extreme = torch.sum(torch.abs(pred_extreme-gt_extreme)).item()*low_max/gt_nta.size(0)/3751
                                            
        losses.append([loss_nta, loss_tract, loss_block, loss_extreme])

        # ----------------
        # bootstrap
        # ----------------
        n = 10000
        bs_errors = []
        pred_high = pred_extreme.detach().cpu().numpy()
        gt_high = gt_extreme.detach().cpu().numpy()
        for i in tqdm(range(n)):
            index = np.random.randint(0,len(pred_high),len(pred_high))
            bs_pred = pred_high[index]
            bs_gt = gt_high[index]
            bs_errors.append(np.sum(np.abs(bs_pred-bs_gt))*low_max/pred_high.shape[0]/chunk_size/3751)
        stds.append(np.std(bs_errors))
        
    # ----------------
    # save results
    # ----------------
    all_losses.append(losses)
    
    all_losses = pd.DataFrame(np.concatenate(all_losses),
                              columns=[('puma', 'nta'),
                                       ('puma', 'tract'),
                                       ('puma', 'block'),
                                       ('puma', 'extreme')],
                              index=domains)
    all_losses.to_csv('disaggregation.csv')
    stds = pd.DataFrame(np.array(stds).T, 
                    columns=[f'({low}, {high})'],
                    index=domains)
    stds.to_csv('stds.csv')
    print('Done...')
    
if __name__ == "__main__":
    main()