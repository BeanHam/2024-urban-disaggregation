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
    data_path = parameters['data_path']
    input_dim = parameters['input_dim']
    hidden_dim = parameters['hidden_dim']
    output_dim = parameters['output_dim']
    batch_size = parameters['batch_size']
    epochs = parameters['epochs']
    learning_rate = parameters['learning_rate']
    learning_rate_step = parameters['learning_rate_step']
    learning_rate_ratio = parameters['learning_rate_ratio']
    epoch_check = parameters['epoch_check']
    log = parameters['log']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ################### load data #################
    print('Load Datasets...')
    
    att_low_res_path = data_path+'attributes/'+low_res_name+'.npy'
    adj_low_res_path = data_path+'adjacencies/'+low_res_name+'.npy'
    att_super_res_path = data_path+'attributes/'+super_res_name+'.npy'
    adj_super_res_path = data_path+'adjacencies/'+super_res_name+'.npy'
    linkage_path = data_path+'linkages/'+low_res_name+'_'+super_res_name+'.npy'
    dataset_train, dataset_val, dataset_test = load_data(att_low_res_path, 
                                                         adj_low_res_path, 
                                                         att_super_res_path, 
                                                         adj_super_res_path,
                                                         linkage_path)
    
    ################### initiate model, optimizer, criterion, and scheduler #################
    print('Initialize Model...')
    model = GraphSR(input_dim, 
                    hidden_dim, 
                    output_dim,
                    dataset_train.linkage.to(device)).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.L1Loss().to(device)
    scheduler = StepLR(optimizer, learning_rate_step, learning_rate_ratio)
    
    ################### load from previous training
    if log: 
        model.load_state_dict(torch.load(f'model_state/graphSR_{low_res_name}_{super_res_name}'))
        loss_track = np.load(f'logs/loss_track_{low_res_name}_{super_res_name}.npy')
    else:
        loss_track = []
    
    ################### training & evaluation #################
    print('Training Model...')
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