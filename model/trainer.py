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
    opt = parser.parse_args()
    
    ################### unload parameters #################
    parameter_path = opt.parameter_path.lower()
    with open(parameter_path) as json_file:
        parameters = json.load(json_file)
        
    low_res_name = parameters['low_res_name']
    super_res_name = parameters['super_res_name']
    att_path = parameters['att_path']
    adj_path = parameters['adj_path']
    batch_size = parameters['batch_size']
    epochs = parameters['epochs']
    learning_rate = parameters['learning_rate']
    learning_rate_step = parameters['learning_rate_step']
    learning_rate_ratio = parameters['learning_rate_ratio']
    attention = parameters['attention']
    epoch_check = parameters['epoch_check']
    log = parameters['log']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ################### load data #################
    print('Load Datasets...')
    
    att_low_res_path = att_path+low_res_name+'.npy'
    adj_low_res_path = adj_path+low_res_name+'.npy'
    att_super_res_path = att_path+super_res_name+'.npy'
    adj_super_res_path = adj_path+super_res_name+'.npy'
    dataset_train, dataset_val, dataset_test = load_data(att_low_res_path, 
                                                         adj_low_res_path, 
                                                         att_super_res_path, 
                                                         adj_super_res_path)
    
    ################### initiate model, optimizer, criterion, and scheduler #################
    print('Initialize Model...')
    model = GraphSR(dataset_train.adj_super.size(0), attention).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.L1Loss().to(device)
    scheduler = StepLR(optimizer, learning_rate_step, learning_rate_ratio)
    
    
    ################### load from previous training
    if log: model.load_state_dict(torch.load(f'model_state/graphSR_{low_res_name}_{super_res_name}'))
    
    
    ################### training & evaluation #################
    print('Training Model...')
    loss_track = []
    for epoch in tqdm(range(epochs)):
        train(model, criterion, optimizer, scheduler, device, batch_size, dataset_train)
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