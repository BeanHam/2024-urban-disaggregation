import os
import json
import time
import numpy as np
import pandas as pd
import geojson
import itertools
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import linemerge, unary_union, polygonize, cascaded_union
from utils import *


def main():
    
    print(f"Load configuration...")
    with open('parameters.json') as json_file:
        parameters = json.load(json_file)
        
    ## get boundary
    print(f"Prepare boundary...")
    tract_root = parameters['tract']
    taxi_root = parameters['taxi']
    nta_root = parameters['nta']
    puma_root = parameters['puma']
    boundary = get_boundary(taxi_root, parameters)
    
    ## get geodata
    print(f"Prepare geodata...")
    geodata_puma, A_puma = get_adjacency(puma_root, boundary, 'puma')
    geodata_nta, A_nta = get_adjacency(nta_root, boundary, 'nta')
    geodata_taxi, A_taxi = get_adjacency(taxi_root, boundary, 'taxi')
    geodata_tract, A_tract = get_adjacency(tract_root, boundary, 'tract')
    
    ## get linkage
    puma_nta_linkage = get_linkage(geodata_puma, geodata_nta, "puma_nta")
    nta_taxi_linkage = get_linkage(geodata_nta, geodata_taxi, "nta_taxi")
    taxi_tract_linkage = get_linkage(geodata_taxi, geodata_tract, "taxi_tract")
    nta_tract_linkage = get_linkage(geodata_nta, geodata_tract, "nta_tract")
    
    ## get attributes
    print(f"Prepare attributes...")
    get_attributes(parameters, geodata_puma, geodata_nta, geodata_taxi, geodata_tract)
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    