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
    parser.add_argument('parameter_path')
    opt = parser.parse_args()
    
    ################### unload parameters #################
    parameter_path = opt.parameter_path.lower()
    with open(parameter_path) as json_file:
        parameters = json.load(json_file)
    data_path = parameters['data_path']
    batch_size = parameters['batch_size']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ################### inference
    print('Inferencing...')
    names = ['puma', 'nta', 'tract', 'block']
    losses = []
    for low_res_name, super_res_name in combinations(names, 2):

        ## load data
        dataset_train, dataset_val, dataset_test, X_max = load_data(low_res_name,
                                                                    super_res_name,
                                                                    parameters)
        
        ## load model
        model = GraphSR(dataset_train.linkage.to(device)).to(device)
        criterion = nn.L1Loss().to(device)
        model.load_state_dict(torch.load(f'model_state/graphSR_{low_res_name}_{super_res_name}'))
        
        ## pred
        loss, pred_super, gt_super, gt_low = evaluation(model, criterion, device, batch_size, dataset_test)
        losses.append(loss*X_max)
        
        ## load geodata
        gt_super = gt_super.cpu()
        pred_super = pred_super.cpu()
        geodata_super_res = pd.read_csv(f'{data_path}/geodata/{super_res_name}.csv')        
        geodata_super_res['count_gt'] = torch.mean(gt_super.detach()*X_max, dim=0).numpy()
        geodata_super_res['count_pred'] = torch.mean(pred_super.detach()*X_max, dim=0).numpy()
        geodata_super_res['diff'] = (torch.mean((pred_super - gt_super),dim=0)*X_max).detach().numpy()
        geodata_super_res.to_csv(f'inferences/vis_super_{low_res_name}_{super_res_name}.csv', header=True)
        
    losses = pd.DataFrame(np.array(losses).reshape(1,-1),
                          columns=['puma_nta', 'puma_tract', 'puma_block',
                                   'nta_tract', 'nta_block', 'tract_block'])
    losses.to_csv(f'inferences/results.csv')
    
    print('Done...')
if __name__ == "__main__":
    main()