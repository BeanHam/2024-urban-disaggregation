import numpy as np
import json
import torch
import argparse
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
        
    root = parameters['root']
    batch_size = parameters['batch_size']
    epochs = parameters['epochs']
    learning_rate = parameters['learning_rate']
    learning_rate_step = parameters['learning_rate_step']
    learning_rate_ratio = parameters['learning_rate_ratio']
    attention = parameters['attention']
    epoch_check = parameters['epoch_check']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ################### load data #################
    print('Load Datasets...')
    dataset_train, dataset_val, dataset_test = load_data(root)
    
    
    ################### initiate model, optimizer, criterion, and scheduler #################
    print('Initialize Model...')
    model = GraphSR(attention).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.L1Loss().to(device)
    scheduler = StepLR(optimizer, learning_rate_step, learning_rate_ratio)
    
    ################### training & evaluation #################
    print('Training Model...')
    loss_track = []
    for epoch in tqdm(range(epochs)):
        train(model, criterion, optimizer, scheduler, device, batch_size, dataset_train)
        loss,_,_,_ = evaluation(model, criterion, device, batch_size, dataset_val)
        loss_track.append(loss)
        if epoch % epoch_check == 0:
            print(f'Validation Loss: {round(loss, 4)}')
    
    ################### save results #################
    print('Done Training...')
    torch.save(model.state_dict(), f'model_state/graphSR_attention_{attention}')
    np.save(f'logs/loss_track_attention_{attention}.npy', loss_track)
    
if __name__ == "__main__":
    main()