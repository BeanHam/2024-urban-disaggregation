import numpy as np
import json
import torch
import argparse
import os
from itertools import combinations
from utils import *
from models import *

## seed
seed = 816
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main():
    
    ## parameters
    print('Load Parameters...')
    parser = argparse.ArgumentParser()
    parser.add_argument('parameter_path')
    opt = parser.parse_args()
    
    ## unload parameters
    parameter_path = opt.parameter_path.lower()
    with open(parameter_path) as json_file:
        parameters = json.load(json_file)
    batch_size = parameters['batch_size']
    learning_rate = parameters['learning_rate']
    epochs = parameters['epochs']
    epoch_check = parameters['epoch_check']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = ['puma', 'nta', 'tract', 'block']

    ## iterate all combinations
    for low_res_name, super_res_name in combinations(names, 2):

        ## load data
        print('===================================')
        print(f'{low_res_name} -> {super_res_name}...')
        print('===================================')
        print('Load Datasets...')
        dataset_train, dataset_val, _, _ = load_data(low_res_name, super_res_name, parameters)
        linkage = dataset_train.linkage

        ## initiate model
        print('Initialize Model...')
        model = GraphSR(linkage).to(device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        criterion = nn.L1Loss().to(device)
        loss_track = []

        ## training & evaluation
        print('Training Model...')
        loss_min = np.inf
        counter = 0
        for epoch in tqdm(range(epochs)):

            ## training
            train(model, criterion, optimizer, device, batch_size, dataset_train)

            ## evaluation
            loss,_,_,_ = evaluation(model, criterion, device, batch_size, dataset_val)

            ### early stopping: tolerance = 10
            if counter >= 10:
                ## early stop
                print(f" Early Stopping at Epoch {epoch}")
                print(f' Validation Loss: {round(loss, 5)}')
                break
            
            ## check min loss
            if loss < loss_min:

                ## save loss min
                loss_min = loss
                loss_track.append(loss)

                ## save results every epoch
                torch.save(model.state_dict(), f'model_state/graphSR_{low_res_name}_{super_res_name}')
                np.save(f'logs/loss_track_{low_res_name}_{super_res_name}.npy', loss_track)

                ## validation check
                if epoch % epoch_check == 0:
                    print(f' Validation Loss: {round(loss, 5)}')
            else:
                counter += 1
                
        print('Done Training...')
        print('                ')

if __name__ == "__main__":
    main()