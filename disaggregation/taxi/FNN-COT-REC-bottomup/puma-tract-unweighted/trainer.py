import numpy as np
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from itertools import combinations
from utils import *
from models import *

def main():
    
    # ----------------
    # Load Parameters
    # ----------------
    with open('config.json') as json_file:
        parameters = json.load(json_file)
        
    root = parameters['root']
    data_name = parameters['data_name']
    epochs = parameters['epochs']
    batch_size = parameters['batch_size']
    epoch_check = parameters['epoch_check']    
    learning_rate = parameters['learning_rate']
    tolerence = parameters['tolerence']
    dims = parameters['dims']
    parameters['source_path'] = root+data_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    low, high = 'puma', 'tract'
    
    # ----------------
    # Load Data
    # ----------------
    print('=======================')
    print(f'{low} -> {high}...')
    print('   ---Load Datasets...')
    dataset_train, dataset_val, _, _, linkages = load_data(low, high, parameters)
    low_size = dims[low]
    high_size = dims[high]
    hidden_sizes = [dims['nta']]
    linkages = [linkage.to(device) for linkage in linkages]
    
    # ----------------
    # Initialize Model
    # ----------------
    print('   ---Initialize Model...')
    model = DisAgg(low_size, high_size, hidden_sizes, linkages).to(device)
    model.apply(weights_init)
    es = EarlyStopping(model, tolerance=tolerence)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss().to(device)
    loss_track_puma = []
    loss_track_nta = []
    loss_track_tract = []
    loss_track_puma_rec = []
    loss_track_nta_rec = []
    
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
              criterion, 
              optimizer,
              device, 
              batch_size, 
              dataset_train)
        
        # ----------------
        # Evaluation
        # ----------------
        model.eval()
        eval_results = evaluation(model, 
                                  criterion, 
                                  device, 
                                  batch_size, 
                                  dataset_val)
        loss_track_puma.append(eval_results['loss_puma'])
        loss_track_nta.append(eval_results['loss_nta'])
        loss_track_tract.append(eval_results['loss_tract'])
        loss_track_puma_rec.append(eval_results['loss_puma_rec'])
        loss_track_nta_rec.append(eval_results['loss_nta_rec'])
        loss = eval_results['loss_tract']
        
        # ----------------
        # Early Stop Check
        # ----------------
        es(loss)            
        if es.early_stop:
            print(f" Early Stopping at Epoch {epoch}")
            print(f' Validation Loss: {round(loss*1000, 5)}')
            break                
        if es.save_model:            
            torch.save(model.state_dict(), f'model_state/{low}_{high}')
            np.save(f'logs/puma_loss.npy', loss_track_puma)
            np.save(f'logs/nta_loss.npy', loss_track_nta)
            np.save(f'logs/tract_loss.npy', loss_track_tract)
            np.save(f'logs/puma_rec_loss.npy', loss_track_puma_rec)
            np.save(f'logs/nta_rec_loss.npy', loss_track_nta_rec)
        
        # ----------------
        # Progress Check
        # ----------------
        if epoch % epoch_check == 0: print(f' Validation Loss: {round(loss*1000, 5)}')
                
    print('Done Training...')
    print('                ')

if __name__ == "__main__":
    main()