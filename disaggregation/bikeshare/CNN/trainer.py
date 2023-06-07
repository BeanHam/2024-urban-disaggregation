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
        
        # ----------------
        # Initialize Model
        # ----------------
        print('   ---Initialize Model...')
        model = DisAgg().to(device)
        es = EarlyStopping(model, tolerance=tolerence)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.L1Loss().to(device)
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
                  criterion, 
                  optimizer,
                  device, 
                  batch_size, 
                  dataset_train)
            
            # ----------------
            # Evaluation
            # ----------------
            model.eval()
            loss,_,_ = evaluation(model, 
                                  criterion, 
                                  device, 
                                  batch_size, 
                                  dataset_val)
            loss_track.append(loss)
            
            # ----------------
            # Early Stop Check
            # ----------------
            es(loss)            
            if es.early_stop:
                print(f" Early Stopping at Epoch {epoch}")
                print(f' Validation Loss: {round(loss, 5)}')
                break                
            if es.save_model:            
                torch.save(model.state_dict(), f'model_state/{low}_{high}')
                np.save(f'logs/{low}_{high}.npy', loss_track)
            
            # ----------------
            # Progress Check
            # ----------------
            if epoch % epoch_check == 0: print(f' Validation Loss: {round(loss, 5)}')
                    
        print('Done Training...')
        print('                ')

if __name__ == "__main__":
    main()