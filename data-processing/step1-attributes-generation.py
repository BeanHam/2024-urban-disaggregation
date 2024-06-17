import json
import argparse
from utils import *

def main():   
        
    #-------------------------
    # arguments
    #-------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='filename of test data')
    args = parser.parse_args()
    data_name = args.data
       
    #--------------------
    # Parameters
    #--------------------
    print(f"Load configuration...")      
    with open('parameters.json') as json_file:
        parameters = json.load(json_file)
    if data_name == 'taxi':
        parameters['root']="D:/all-data/nyc-taxi/raw_chunk_data/"
        parameters['year']="2016"
    elif data_name == 'bikeshare': 
        parameters['root']="D:/all-data/nyc-bikeshare/raw_data/"
        parameters['year']="2021"
    else: 
        parameters['root']="D:/all-data/nyc-911/911.csv"
        parameters['year']="2021"
    
    
    print(f"Prepare boundary...")
    puma_root = parameters['puma']
    nta_root = parameters['nta']
    tract_root = parameters['tract']
    block_root = parameters['block']
    #extreme_root = parameters['extreme']
    boundary_root = parameters['boundary']
    boundary = get_boundary(boundary_root, parameters)
    
    #--------------------
    # Load Geodatas
    #--------------------
    print(f"Prepare geodata...")
    geodata_puma = get_geodata(puma_root, boundary, data_name, 'puma')
    geodata_nta = get_geodata(nta_root, boundary, data_name, 'nta')
    geodata_tract = get_geodata(tract_root, boundary, data_name, 'tract')
    geodata_block = get_geodata(block_root, boundary, data_name, 'block')
    #get_extreme_data(geodata_block, data_name)
    #geodata_extreme = get_geodata(extreme_root, boundary, data_name, 'extreme')
        
    #--------------------
    # Get Linkages
    #--------------------
    print(f"Prepare linkage...")
    puma_nta_linkage = get_linkage(geodata_puma, geodata_nta, data_name, "puma_nta")
    puma_tract_linkage = get_linkage(geodata_puma, geodata_tract, data_name, "puma_tract")
    puma_block_linkage = get_linkage(geodata_puma, geodata_block, data_name, "puma_block")
    #puma_extreme_linkage = get_linkage(geodata_puma, geodata_extreme, data_name, "puma_extreme")
    nta_tract_linkage = get_linkage(geodata_nta, geodata_tract, data_name, "nta_tract")
    nta_block_linkage = get_linkage(geodata_nta, geodata_block, data_name, "nta_block")
    #nta_extreme_linkage = get_linkage(geodata_nta, geodata_extreme, data_name, "nta_extreme")
    tract_block_linkage = get_linkage(geodata_tract, geodata_block, data_name, "tract_block")
    #tract_extreme_linkage = get_linkage(geodata_tract, geodata_extreme, data_name, "tract_extreme")
    #block_extreme_linkage = get_linkage(geodata_block, geodata_extreme, data_name, "block_extreme")  
    
    #--------------------
    # Get Attributes
    #--------------------
    print(f"Prepare attributes...")
    if data_name == 'taxi': extraction_function = get_attributes_taxi
    elif data_name =='bikeshare': extraction_function = get_attributes_bikeshare
    else: extraction_function = get_attributes_911
    extraction_function(parameters,
                        geodata_puma, 
                        geodata_nta,
                        geodata_tract, 
                        geodata_block)
    
if __name__ == "__main__":
    main()
    