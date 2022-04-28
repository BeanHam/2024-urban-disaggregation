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
        
    ## get geodata
    print(f"Prepare geodata...")
    geodata_low_resolution, \
    geodata_super_resolution, \
    adjacency_low_resolution, \
    adjacency_super_resolution = get_geodata(parameters)
    
    ## get node attributes
    print(f"Prepare node attributes...")
    node_attributes_low_resolution, \
    node_attributes_super_resolution = get_attribute_data(parameters,
                                                          geodata_low_resolution, 
                                                          geodata_super_resolution)
    
    ## save data
    np.save('X_low_res.npy', node_attributes_low_resolution)
    np.save('geodata_low_res.npy', geodata_low_resolution.values)
    np.save('A_res.npy', adjacency_low_resolution)
    
    np.save('X_super_res.npy', node_attributes_super_resolution)
    np.save('geodata_super_res.npy', geodata_super_resolution.values)
    np.save('A_res.npy', adjacency_super_resolution)
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    