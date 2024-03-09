import json
import argparse
from utils import *

def main():   
        
    #-------------------------
    # arguments
    #-------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', required=True, help='filename of test data')
    args = parser.parse_args()
    data_name = args.data_name
       
    #--------------------
    # Parameters
    #--------------------
    print(f"Load configuration...")
    if data_name in ['taxi', 'bikeshare', '911-call']: 
        parameter_file = 'parameters.json'
    else: 
        parameter_file = 'chicago-parameters.json'        
    with open(parameter_file) as json_file:
        parameters = json.load(json_file)
        
    print(f"Prepare boundary & extreme data...")
    community_root = parameters['community']
    tract_root = parameters['tract']
    block_root = parameters['block']
    extreme_root = parameters['extreme']
    boundary_root = parameters['boundary']
    boundary = get_boundary(boundary_root, parameters)
    
    #--------------------
    # Load Geodatas
    #--------------------
    print(f"Prepare geodata...")
    geodata_community = get_geodata(community_root, boundary, data_name, 'community')
    geodata_tract = get_geodata(tract_root, boundary, data_name, 'tract')
    geodata_block = get_geodata(block_root, boundary, data_name, 'block')
    get_extreme_data(geodata_block, data_name, n_splits=2)
    geodata_extreme = get_geodata(extreme_root, boundary, data_name, 'extreme')
    
    #--------------------
    # Get Linkages
    #--------------------
    print(f"Prepare linkage...")
    community_tract_linkage = get_linkage(geodata_community, geodata_tract, data_name, "community_tract")
    community_block_linkage = get_linkage(geodata_community, geodata_block, data_name, "community_block")
    community_extreme_linkage = get_linkage(geodata_community, geodata_extreme, data_name, "community_extreme")
    tract_block_linkage = get_linkage(geodata_tract, geodata_block, data_name, "tract_block")
    tract_block_linkage = get_linkage(geodata_tract, geodata_extreme, data_name, "tract_extreme")
    block_extreme_linkage = get_linkage(geodata_block, geodata_extreme, data_name, "block_extreme") 
    
    #--------------------
    # Get Attributes
    #--------------------
    print(f"Prepare attributes...")
    get_attributes_chicago(parameters, 
                           geodata_community,
                           geodata_tract, 
                           geodata_block,
                           geodata_extreme)
    
if __name__ == "__main__":
    main()
    