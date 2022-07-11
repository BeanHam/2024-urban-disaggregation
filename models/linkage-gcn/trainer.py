import numpy as np
import json
import torch
import argparse
import os 
from utils import *
from models import *


def main():
    
    ################# parameters #################
    print('Load Parameters...')
    parser = argparse.ArgumentParser()
    parser.add_argument('parameter_path', help='path to the configuration of the model')
    parser.add_argument('low_res_name')
    parser.add_argument('super_res_name')
    opt = parser.parse_args()
    
    ################# unload parameters ##########
    parameter_path = opt.parameter_path.lower()
    low_res_name = opt.low_res_name.lower()
    super_res_name = opt.super_res_name.lower()
    with open(parameter_path) as json_file:
        parameters = json.load(json_file)
    
    batch_size = parameters['batch_size']
    learning_rate = parameters['learning_rate']
    epochs = parameters['epochs']
    epoch_check = parameters['epoch_check']
    log = parameters['log']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ################### load data #################
    print('Load Datasets...')
    dataset_train, dataset_val, dataset_test, X_max = load_data(low_res_name,
                                                                super_res_name,
                                                                parameters)
    
    ############### initiate model #################
    print('Initialize Model...')
    model = GraphSR(dataset_train.linkage.to(device)).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.L1Loss().to(device)
    
    ################### load from previous training
    if log: 
        model.load_state_dict(torch.load(f'model_state/graphSR_{low_res_name}_{super_res_name}'))
        loss_track = np.load(f'logs/loss_track_{low_res_name}_{super_res_name}.npy')
    else:
        loss_track = []
    
    ################### training & evaluation #################
    print('Training Model...')
    for epoch in tqdm(range(epochs)):
        train(model, criterion, optimizer, device, batch_size, dataset_train)
        loss,_,_,_ = evaluation(model, criterion, device, batch_size, dataset_val)
        
        ## save loss
        loss_track.append(loss)
        
        ## save results every epoch
        torch.save(model.state_dict(), f'model_state/graphSR_{low_res_name}_{super_res_name}')
        np.save(f'logs/loss_track_{low_res_name}_{super_res_name}.npy', loss_track)
        
        ## validation check
        if epoch % epoch_check == 0:
            print(f'Validation Loss: {round(loss, 4)}')
            
    print('Done Training...')
    
if __name__ == "__main__":
    main()