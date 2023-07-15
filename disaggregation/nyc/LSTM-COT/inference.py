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
    resolutions = [('puma', 'nta'), 
                   ('puma', 'tract'),
                   ('puma', 'block')]#,
                   #('puma', 'extreme')]
    domains = ['taxi', 'bikeshare', '911-call']
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
        parameters['high'] = high
        
        # ----------------
        # Initialize Model
        # ----------------
        print('   ---Initialize Model...')
        if high == 'nta':
            hidden_sizes = [int(0.5*(low_size+high_size))]
            model = DisAgg_puma_nta(low_size, high_size, hidden_sizes).to(device)        
        elif high == 'tract':
            hidden_sizes = [dims['nta']]
            model = DisAgg_puma_tract(low_size, high_size, hidden_sizes).to(device)
        elif high == 'block':
            hidden_sizes = [dims['nta'], dims['tract']]
            model = DisAgg_puma_block(low_size, high_size, hidden_sizes).to(device)
        else:
            hidden_sizes = [dims['nta'], dims['tract'], dims['block']]
            model = DisAgg_puma_extreme(low_size, high_size, hidden_sizes).to(device)
                
        model.load_state_dict(torch.load(f'model_state/{data}_{low}_{high}'))
        model.eval()        
        
        # ----------------
        # target data prediction
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
                                      device, 
                                      batch_size, 
                                      dataset_test,
                                      parameters)
            pred_high = eval_results['pred']
            gt_high = eval_results['gt']
            losses.append(torch.mean(torch.sum(torch.abs(pred_high-gt_high), dim=1)/dims['block']).item()*low_max)
            
            # ----------------
            # save
            # ----------------
            if domain == data:
                pred_high = pred_high.detach().cpu().numpy()
                np.save(f'../../../results/pred-gt/{data}/LSTM_COT_{low}_{high}_pred.npy', pred_high)
        
        all_losses.append(losses)
        
    # ----------------
    # save results
    # ----------------
    all_losses = pd.DataFrame(
        np.array(all_losses), 
        index=resolutions,
        columns=domains
    )
    all_losses.to_csv(f'{data}_results.csv')
    print('Done...')
    
if __name__ == "__main__":
    main()