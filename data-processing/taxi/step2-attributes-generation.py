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
    puma_root = parameters['puma']
    nta_root = parameters['nta']
    taxi_root = parameters['taxi']
    tract_root = parameters['tract']
    block_root = parameters['block']
    extreme_root = parameters['extreme']
    boundary = get_boundary(taxi_root, parameters)
    
    #--------------------
    # Load Geodatas
    #--------------------
    print(f"Prepare geodata...")
    geodata_puma = get_geodata(puma_root, boundary, 'puma')
    geodata_nta = get_geodata(nta_root, boundary, 'nta')
    geodata_tract = get_geodata(tract_root, boundary, 'tract')
    geodata_block = get_geodata(block_root, boundary, 'block')
    geodata_extreme = get_geodata(extreme_root, boundary, 'extreme')
    
    #--------------------
    # Get Linkages
    #--------------------
    print(f"Prepare linkage...")
    puma_nta_linkage = get_linkage(geodata_puma, geodata_nta, "puma_nta")
    puma_tract_linkage = get_linkage(geodata_puma, geodata_tract, "puma_tract")
    puma_block_linkage = get_linkage(geodata_puma, geodata_block, "puma_block")
    puma_extreme_linkage = get_linkage(geodata_puma, geodata_extreme, "puma_extreme")
    nta_tract_linkage = get_linkage(geodata_nta, geodata_tract, "nta_tract")
    nta_block_linkage = get_linkage(geodata_nta, geodata_block, "nta_block")
    nta_extreme_linkage = get_linkage(geodata_nta, geodata_extreme, "nta_extreme")
    tract_block_linkage = get_linkage(geodata_tract, geodata_block, "tract_block")
    tract_extreme_linkage = get_linkage(geodata_tract, geodata_extreme, "tract_extreme")
    block_extreme_linkage = get_linkage(geodata_block, geodata_extreme, "block_extreme")  
    
    #--------------------
    # Get Attributes
    #--------------------
    print(f"Prepare attributes...")
    get_attributes(parameters, 
                   geodata_puma, 
                   geodata_nta,
                   geodata_tract, 
                   geodata_block,
                   geodata_extreme)
    
if __name__ == "__main__":
    main()
    