import json
from utils import *

def main():
    
    print(f"Load configuration...")
    with open('parameters.json') as json_file:
        parameters = json.load(json_file)
        
    #--------------------
    # Parameters
    #--------------------
    print(f"Prepare boundary...")
    root = parameters['boundary-root']
    community_root = parameters['community']
    tract_root = parameters['tract']
    block_root = parameters['block']
    extreme_root = parameters['extreme']
    boundary = get_boundary(root, parameters)
    
    #--------------------
    # Load Geodatas
    #--------------------
    print(f"Prepare geodata...")
    geodata_community = get_geodata(community_root, boundary, 'community')
    geodata_tract = get_geodata(tract_root, boundary, 'tract')
    geodata_block = get_geodata(block_root, boundary, 'block')
    geodata_extreme = get_geodata(extreme_root, boundary, 'extreme')
    
    #--------------------
    # Get Linkages
    #--------------------
    print(f"Prepare linkage...")
    community_tract_linkage = get_linkage(geodata_community, geodata_tract, "community_tract")
    community_block_linkage = get_linkage(geodata_community, geodata_block, "community_block")
    community_extreme_linkage = get_linkage(geodata_community, geodata_extreme, "community_extreme")
    tract_block_linkage = get_linkage(geodata_tract, geodata_block, "tract_block")
    tract_block_linkage = get_linkage(geodata_tract, geodata_extreme, "tract_extreme")
    block_extreme_linkage = get_linkage(geodata_block, geodata_extreme, "block_extreme") 
    
    #--------------------
    # Get Attributes
    #--------------------
    print(f"Prepare attributes...")
    get_attributes(parameters, 
                   geodata_community,
                   geodata_tract, 
                   geodata_block,
                   geodata_extreme)
    
if __name__ == "__main__":
    main()
    