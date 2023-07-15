import numpy as np
import pandas as pd
import argparse
import json
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
    factor = parameters['factor']
    batch_size = parameters['batch_size']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolutions = [('com', 'tract'),
                   ('com', 'block'),
                   ('com', 'extreme')]
    all_losses = []
    
    # ----------------
    # Inference
    # ----------------
    print('Inferencing...')
    for low, high in resolutions:
            
        print('   =======================')
        print(f'   {low} -> {high}...')
        losses = []
                   
        # -----------------
        # Load Source Model
        # -----------------
        model = DisAgg().to(device)
        criterion = nn.L1Loss().to(device)
        model.load_state_dict(torch.load(f'model_state/{low}_{high}'))
        model.eval()
        
        # ----------------
        # Load Parameters
        # ----------------
        source_path = root+data
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
        losses.append(np.mean(np.sum(np.abs(pred_all*low_max-test), dim=1)/factor))
        
        # ----------------
        # save results
        # ----------------
        all_losses.append(losses)
    
    all_losses = pd.DataFrame(
        np.stack(all_losses), 
        index=resolutions
    )
    all_losses.to_csv(f'{data}_results.csv')
    print('Done...')
    
if __name__ == "__main__":
    main()