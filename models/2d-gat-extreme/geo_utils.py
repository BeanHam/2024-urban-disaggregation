import os
import json
import time
import numpy as np
import pandas as pd
import geojson
import itertools
import geopandas
import matplotlib.pyplot as plt
from shapely import wkt
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import linemerge, unary_union, polygonize, cascaded_union

import warnings
#from shapely.errors import ShapelyDeprecationWarning
#warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

def change_time_format(time):
    
    """
    This function is used to change time format.
    """
    if 'PM' in time: return str(int(time[:2])+12)
    else: return time[:2]
    
def get_boundary(root, parameters):
    
    """
    This function is used to generate the Manhanttan boundary
    
    """
    
    ## load data & zones
    data = pd.read_csv(root)
    zones = parameters['zones']
    
    ## make the data -> geodata
    gdf = geopandas.read_file(root)
    gdf.crs = 'epsg:4326'
    data['the_geom'] = data['the_geom'].apply(wkt.loads)
    gdf = geopandas.GeoDataFrame(data, crs='epsg:4326')
    gdf = gdf[np.isin(gdf.OBJECTID, zones)]
    
    ## generate boundary
    new_polygons = []
    for geom in gdf.the_geom:
        multi_polygons = list(geom)
        areas = [poly.area for poly in multi_polygons]
        new_polygons.append(multi_polygons[np.where(areas == np.max(areas))[0][0]])
    gdf.the_geom = new_polygons
    polygons = [i for i in gdf.the_geom.values]
    boundary = geopandas.GeoSeries(unary_union(polygons))[0]
    
    return boundary


def get_adjacency(root, boundary, data_name):
    
    ## load data & make geodata
    data = pd.read_csv(root)
    gdf = geopandas.read_file(root)
    gdf.crs = 'epsg:4326'
    data['the_geom'] = data['the_geom'].apply(wkt.loads)
    gdf = geopandas.GeoDataFrame(data, crs='epsg:4326')
    
    ## subset regions within boundary
    geodata = []
    for geom in gdf.the_geom:
        multi_polygons = list(geom)
        for polygon in multi_polygons:
            if polygon.intersects(boundary):
                area = polygon.area*10**11
                geodata.append([polygon, area])
    geodata = pd.DataFrame(geodata, columns=['geometry', 'shape_area'])
    
    ## generate adjacency matrix
    A = np.zeros((geodata.shape[0], geodata.shape[0]))
    for i in range(len(geodata)):
        for j in range(len(geodata)):
            A[i,j] = geodata.geometry[i].intersects(geodata.geometry[j])
    
    ## save  
    geodata.to_csv(f'geodata/{data_name}.csv',header=True)
    np.save(f'adjacencies/{data_name}.npy', A)
    
    return geodata, A



def get_linkage(low_res_geodata, super_res_geodata, linkage_name):
    
    """
    
    This function is used to get the linkage between low resolution graph to
    super resolution graph using geodata.
    
    
    """
    linkage = np.zeros((low_res_geodata.shape[0], super_res_geodata.shape[0]))
    for i in range(len(low_res_geodata)):
        for j in range(len(super_res_geodata)):
            linkage[i,j] = low_res_geodata.geometry[i].contains(super_res_geodata.geometry[j].centroid)
       
    ## save  
    np.save(f'linkages/{linkage_name}.npy', linkage)
    
    return linkage