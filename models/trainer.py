import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
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
        
    dims = parameters['dims']
    root = parameters['root']
    epochs = parameters['epochs']
    tolerence = parameters['tolerence']
    batch_size = parameters['batch_size']
    epoch_check = parameters['epoch_check']
    learning_rate = parameters['learning_rate']
    parameters['path'] = root+data
    parameters['rec'] = rec
    parameters['local'] = local
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolutions = [#('puma', 'nta'),]
                   #('puma', 'tract'),]
                   ('puma', 'block')]
    print('=======================')
    print(f'---Dataset: {data}')
    print(f'---REC: {rec}')
    print(f'---LOCAL: {local}')
    
    # -------------------
    # Iterate Resolutions
    # -------------------
    for coarse, fine in resolutions:
        
        # ----------------
        # Load Data
        # ----------------
        print('=======================')
        print(f'{coarse} -> {fine}...')
        print('   ---Load Datasets...')
        parameters['coarse'] = coarse
        parameters['fine'] = fine
        parameters['low'] = dims[coarse]
        parameters['high'] = dims[fine]
        learning_rate_factor = parameters['learning_rate_factor'][data][fine]
        dataset_train, dataset_val, _, _, linkages = load_data(parameters)

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
        
        # ----------------
        # Initialize Opt
        # ----------------
        es = EarlyStopping(model, tolerance=tolerence)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                               'min', 
                                                               factor=learning_rate_factor, 
                                                               #min_lr=1e-5, 
                                                               patience=25)
        print(f'   --- Model Size: {sum(p.numel() for p in model.parameters())}')
        loss_track = []
        
        # -----------------
        # Continue Training
        # -----------------
        if os.path.exists(f'model_state/{data}_{coarse}_{fine}_rec_{rec}_local_{local}'):
            model.load_state_dict(torch.load(f'model_state/{data}_{coarse}_{fine}_rec_{rec}_local_{local}'))
            loss_track = np.load(f'logs/{data}_{coarse}_{fine}_rec_{rec}_local_{local}.npy').tolist()
            es.loss_min = loss_track[-1]
        else:
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
            scheduler.step(loss)

            # ----------------
            # Early Stop Check
            # ----------------
            es(loss)
            if es.early_stop:
                print(f" Early Stopping at Epoch {epoch}")
                print(f' Validation Loss: {round(loss_track[-1]*1000, 5)}')
                break
            if es.save_model:
                loss_track.append(loss)
                torch.save(model.state_dict(), f'model_state/{data}_{coarse}_{fine}_rec_{rec}_local_{local}')
                np.save(f'logs/{data}_{coarse}_{fine}_rec_{rec}_local_{local}.npy', loss_track)
            
            # ----------------
            # Progress Check
            # ----------------
            if epoch % epoch_check == 0:
                print(f' Validation Loss: {round(loss*1000, 5)}. Learning Rate: {scheduler.get_last_lr()[0]}')
            
        print('Done Training...')
        print('                ')

if __name__ == "__main__":
    main()
