import numpy as np
import json
import argparse
import torch
from tqdm import tqdm
from utils import *
from models import *

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
    epochs = parameters['epochs']
    batch_size = parameters['batch_size']
    epoch_check = parameters['epoch_check']    
    learning_rate = parameters['learning_rate']
    tolerence = parameters['tolerence']
    dims = parameters['dims']
    parameters['source_path'] = root+data_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolutions = [('puma', 'nta'), 
                   ('puma', 'tract'),
                   ('puma', 'block'),
                   ('puma', 'extreme')]
    
    # -------------------
    # Iterate Resolutions
    # -------------------
    for low, high in resolutions:
        
        # ----------------
        # Load Data
        # ----------------
        print('=======================')
        print(f'{low} -> {high}...')
        print('   ---Load Datasets...')
        dataset_train, dataset_val, _, _ = load_data(low, high, parameters)
        low_size = dims[low]
        high_size = dims[high]
        
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
        
        model.apply(weights_init)
        es = EarlyStopping(model, tolerance=tolerence)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_track = [] 
        
        # ----------------
        # Iterate Epochs
        # ----------------
        print('   ---Training Model...')
        for epoch in tqdm(range(epochs)):
    
            # ----------------
            # Training
            # ----------------
            model.train()
            train(model,
                  optimizer,
                  device,
                  batch_size,
                  dataset_train,
                  parameters)
            
            # ----------------
            # Evaluation
            # ----------------
            model.eval()
            eval_results = evaluation(model, 
                                      device, 
                                      batch_size, 
                                      dataset_val,
                                      parameters)
            loss = eval_results['loss']
            loss_track.append(loss)
            
            # ----------------
            # Early Stop Check
            # ----------------
            es(loss)            
            if es.early_stop:
                print(f" Early Stopping at Epoch {epoch}")
                print(f' Validation Loss: {round(loss*1000, 5)}')
                break                
            if es.save_model:            
                torch.save(model.state_dict(), f'model_state/{data}_{low}_{high}')
                np.save(f'logs/{data}_{low}_{high}.npy', loss_track)          
            
            # ----------------
            # Progress Check
            # ----------------
            if epoch % epoch_check == 0: print(f' Validation Loss: {round(loss*1000, 5)}')
                    
        print('Done Training...')
        print('                ')

if __name__ == "__main__":
    main()