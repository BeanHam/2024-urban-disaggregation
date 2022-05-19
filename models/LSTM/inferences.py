import numpy as np
import pandas as pd
import json
import argparse
from itertools import combinations
from utils import *
from models import *


def main():
    
    ################# parameters #################
    print('Load Parameters...')
    parser = argparse.ArgumentParser()
    parser.add_argument('parameter_path', help='path to the configuration of the model')
    opt = parser.parse_args()
    
    ################### unload parameters #################
    with open('parameters.json') as json_file:
        parameters = json.load(json_file)
    
    data_path = parameters['data_path']
    batch_size = parameters['batch_size']
    chunk_size = parameters['chunk_size']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ################### inference
    names = ['puma', 'nta', 'taxi', 'tract']
    losses = []
    for low_res_name, super_res_name in combinations(names, 2):

        ## load data
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
        X_super_max = torch.max(torch.from_numpy(np.load(att_super_res_path)).float())
    
        ## load model
        model = GraphSR(input_dim, output_dim).to(device)
        criterion = nn.L1Loss().to(device)
        model.load_state_dict(torch.load(f'model_state/graphSR_{low_res_name}_{super_res_name}'))
        
        ## pred
        loss, pred_super, gt_super, gt_low = evaluation(dataset_val, model, criterion, batch_size, chunk_size, device)
        losses.append((loss*X_super_max).item())
            
        ## load geodata
        geodata_super_res = pd.read_csv(f'{data_path}/geodata/{super_res_name}.csv')
        
        gt_super = gt_super.cpu()
        pred_super = pred_super.cpu()
        
        geodata_super_res['count_gt'] = torch.mean(gt_super.detach()*X_super_max, dim=0).numpy()
        geodata_super_res['count_pred'] = torch.mean(pred_super.detach()*X_super_max, dim=0).numpy()
        geodata_super_res['diff'] = (torch.mean((pred_super - gt_super),dim=0)*X_super_max).detach().numpy()
        geodata_super_res.to_csv(f'visualization/vis_super_{low_res_name}_{super_res_name}.csv', header=True)
        
    losses = pd.DataFrame(np.array(losses).reshape(1,-1),
                          columns=['puma_nta', 'puma_taxi', 'puma_tract', 'nta_taxi', 'nta_tract', 'taxi_tract'])
    losses.to_csv(f'inferences/{low_res_name}_{super_res_name}.csv')
    
if __name__ == "__main__":
    main()