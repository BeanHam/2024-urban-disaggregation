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
    
    ################### unload parameters #################
    parameter_path = opt.parameter_path.lower()
    low_res_name = opt.low_res_name.lower()
    super_res_name = opt.super_res_name.lower()
    with open(parameter_path) as json_file:
        parameters = json.load(json_file)
        
    data_path = parameters['data_path']
    batch_size = parameters['batch_size']
    chunk_size = parameters['chunk_size']
    epochs = parameters['epochs']
    learning_rate = parameters['learning_rate']
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
                                                         linkage_path,
                                                         chunk_size)
    input_dim = dataset_train.adj_low.size(0)
    output_dim = dataset_train.adj_super.size(0)
    
    ################### initiate model, optimizer, criterion, and scheduler #################
    print('Initialize Model...')
    model = GraphSR(input_dim, output_dim).to(device)
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
        train(dataset_train, model, criterion, optimizer, batch_size, chunk_size, device)
        loss,_,_,_ = evaluation(dataset_val, model, criterion, batch_size, chunk_size, device)
        
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