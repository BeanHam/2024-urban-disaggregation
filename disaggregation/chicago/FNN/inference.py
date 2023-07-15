import numpy as np
import pandas as pd
import json
import argparse
from utils import *
from models import *
import warnings
warnings.filterwarnings("ignore")

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
    with open('config.json') as json_file:
        parameters = json.load(json_file)
    
    root = parameters['root']
    dims = parameters['dims']
    batch_size = parameters['batch_size']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolutions = [('com', 'tract'),
                   ('com', 'block'),
                   ('com', 'extreme')]
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
        losses = []
        low_size = dims[low]
        high_size = dims[high]
        
        # ----------------
        # Initialize Model
        # ----------------
        print('   ---Initialize Model...')
        if high == 'tract':             
            hidden_sizes = [int(0.5*(low_size+high_size))]
            model = DisAgg_com_tract(low_size, high_size, hidden_sizes).to(device)        
        elif high == 'block': 
            hidden_sizes = [dims['tract']]
            model = DisAgg_com_block(low_size, high_size, hidden_sizes).to(device)
        else: 
            hidden_sizes = [dims['tract'], dims['block']]
            model = DisAgg_com_extreme(low_size, high_size, hidden_sizes).to(device)
        
        model.load_state_dict(torch.load(f'model_state/{data}_{low}_{high}'))
        model.eval()
                            
        # ----------------
        # Load Target Data
        # ----------------
        parameters['source_path'] = root+data
        _,_,dataset_test, low_max = load_data(low, high, parameters)
        
        # ----------------
        # Prediction
        # ----------------
        _,pred_high, gt_high = evaluation(model, 
                                          device, 
                                          batch_size, 
                                          dataset_test,
                                          parameters)
        losses.append(torch.mean(torch.sum(torch.abs(pred_high-gt_high), dim=1)/dims['block']).item()*low_max)
        all_losses.append(losses)
        
        # ----------------
        # save
        # ----------------
        pred_high = pred_high.detach().cpu().numpy()
        gt_high = gt_high.detach().cpu().numpy()
        np.save(f'../../../results/pred-gt/{data}/FNN_{low}_{high}_pred.npy', pred_high)
        np.save(f'../../../results/pred-gt/{data}/FNN_{low}_{high}_gt.npy', gt_high)
        
    # ----------------
    # save results
    # ----------------
    all_losses = pd.DataFrame(
        np.array(all_losses),
        index=resolutions
    )
    all_losses.to_csv(f'{data}_results.csv')
    print('Done...')
    
if __name__ == "__main__":
    main()