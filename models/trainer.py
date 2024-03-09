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
    parser.add_argument('--nheads', required=True, help='data name')
    parser.add_argument('--cot', required=True, help='whether to do chain-of-training. Values: yes/no')
    parser.add_argument('--rec', required=True, help='which reconstruction to do. Values: no/bottomup/bridge/full')
    args = parser.parse_args()
    data = args.data
    nheads = int(args.nheads)
    cot = args.cot
    rec = args.rec

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
    parameters['cot'] = cot
    parameters['rec'] = rec    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolutions = [('puma', 'nta'),]
                   #('puma', 'tract'),]
                   #('puma', 'block')]
    print('=======================')
    print(f'---Dataset: {data}')
    print(f'---COT: {cot}')
    print(f'---REC: {rec}')
    
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
        parameters['low'] = low
        parameters['high'] = high 
        dataset_train, dataset_val, _, _, linkages = load_data(parameters)
        
        # ----------------
        # Initialize Model
        # ----------------
        print('   ---Initialize Model...')
        if high == 'nta':
            linkages = [             
                linkages['puma_nta'].to(device)
            ]
            hidden_dims = [
                dims['puma'], 
                dims['nta']
            ]
            model = puma_nta(
                hidden_dims, 
                nheads,
                linkages, 
                rec).to(device)
        
        elif high == 'tract':
            linkages = [           
                linkages['puma_nta'].to(device), 
                linkages['puma_tract'].to(device), 
                linkages['nta_tract'].to(device)
            ]            
            hidden_dims = [
                dims['puma'], 
                dims['nta'], 
                dims['tract']
            ]
            model = puma_tract(
                hidden_dims, 
                nheads,
                linkages, 
                rec).to(device)
            
        else:
            linkages = [
                linkages['puma_nta'].to(device), 
                linkages['puma_tract'].to(device), 
                linkages['puma_block'].to(device),                        
                linkages['nta_tract'].to(device), 
                linkages['nta_block'].to(device), 
                linkages['tract_block'].to(device)
            ]            
            hidden_dims = [
                dims['puma'], 
                dims['nta'],
                dims['tract'], 
                dims['block']
            ]
            model = puma_block(
                hidden_dims, 
                nheads,
                linkages, 
                rec).to(device)
                                
        # ----------------
        # Initialize Opt
        # ----------------
        es = EarlyStopping(model, tolerance=tolerence)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, threshold=1e-8, verbose=True)
        print(f'   ---Model Size: {sum(p.numel() for p in model.parameters())}')
        
        # -----------------
        # Continue Training
        # -----------------
        if os.path.exists(f'model_state/{data}_{low}_{high}_cot_{cot}_rec_{rec}_nheads_{nheads}'):
            model.load_state_dict(torch.load(f'model_state/{data}_{low}_{high}_cot_{cot}_rec_{rec}_nheads_{nheads}'))
            loss_track = np.load(f'logs/{data}_{low}_{high}_cot_{cot}_rec_{rec}_nheads_{nheads}.npy').tolist()
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
            loss_track.append(loss)
            scheduler.step(loss)
            
            # ----------------
            # Early Stop Check
            # ----------------
            es(loss)            
            if es.early_stop:
                print(f" Early Stopping at Epoch {epoch}")
                print(f' Validation Loss: {round(loss*1000, 5)}')
                break                
            if es.save_model:     
                torch.save(model.state_dict(), f'model_state/{data}_{low}_{high}_cot_{cot}_rec_{rec}_nheads_{nheads}')
                np.save(f'logs/{data}_{low}_{high}_cot_{cot}_rec_{rec}_nheads_{nheads}.npy', loss_track)
            
            # ----------------
            # Progress Check
            # ----------------
            if epoch % epoch_check == 0: print(f' Validation Loss: {round(loss*1000, 5)}')
                    
        print('Done Training...')
        print('                ')

if __name__ == "__main__":
    main()
