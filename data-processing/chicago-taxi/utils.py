import os
import json
import time
import numpy as np
import pandas as pd
import geojson
import itertools
import geopandas
import haversine as hs
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely import wkt
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import linemerge, unary_union, polygonize, cascaded_union

import warnings
warnings.filterwarnings("ignore") 

def change_time_format(time):
    
    """
    This function is used to change time format.
    """
    if 'PM' in time: return str(int(time[:2])+12)
    else: return time[:2]

#----------------------
# Get Boundary Function
#----------------------
def get_boundary(root, 
                 parameters):
    
    """
    This function is used to generate the Manhanttan boundary
    
    """
    
    ## load data & zones
    data = pd.read_csv(root)
    zones = parameters['zones']
    
    ## make the data -> geodata
    data['geometry'] = data['the_geom'].apply(wkt.loads)
    gdf = geopandas.GeoDataFrame(data, crs='epsg:4326')
    gdf = gdf[np.isin(gdf.DIST_NUM, zones)]
    
    ## generate boundary
    new_polygons = []
    for geom in gdf.geometry:
        multi_polygons = list(geom.geoms)
        areas = [poly.area for poly in multi_polygons]
        new_polygons.append(multi_polygons[np.where(areas == np.max(areas))[0][0]])
    gdf.the_geom = new_polygons
    polygons = [i for i in gdf.the_geom.values]
    boundary = geopandas.GeoSeries(unary_union(polygons))[0]
    
    return boundary

#----------------------
# Get GeoData Function
#----------------------
def get_geodata(root, 
                boundary, 
                data_name):
    
    ## load data & make geodata
    data = pd.read_csv(root)
    data['geometry'] = data['the_geom'].apply(wkt.loads)
    gdf = geopandas.GeoDataFrame(data, crs='epsg:4326')
    
    ## subset regions within boundary
    geodata = []
    for i in tqdm(range(len(gdf))):
        geom = gdf.geometry[i]
        if geom.is_valid == False:
            geom = geom.buffer(0)            
        if geom.type == 'Polygon':
            multi_polygons = [geom]
        else:
            multi_polygons = list(geom.geoms)            
        for polygon in multi_polygons:
            if polygon.intersects(boundary):
                area = polygon.area*10**11
                geodata.append([polygon, area])
    geodata = pd.DataFrame(geodata, columns=['geometry', 'shape_area'])
    
    ## save  
    geodata.to_csv(f'D:/disaggregation-data/chicago-taxi/geodata/{data_name}.csv',header=True)
    
    return geodata

#----------------------
# Get Linkage Function
#----------------------
def get_linkage(low_res_geodata, 
                high_res_geodata, 
                linkage_name):
    
    """
    
    This function is used to get the linkage between low resolution graph to
    super resolution graph using geodata.
    
    
    """
    
    linkage = np.zeros((low_res_geodata.shape[0], high_res_geodata.shape[0]), dtype='int8')
    for i in tqdm(range(len(low_res_geodata))):
        for j in range(len(high_res_geodata)):
            linkage[i,j] = low_res_geodata.geometry[i].contains(high_res_geodata.geometry[j].centroid)

    np.save(f'D:/disaggregation-data/chicago-taxi/linkages/{linkage_name}.npy', linkage)
    
    return linkage

#----------------------
# Count Function
#----------------------
def count(geodata, 
          hourly_data):
    data = geopandas.sjoin(geodata, hourly_data)
    data['index'] = data.index
    data_count = np.zeros(len(geodata))
    data_count[np.unique(data.index)] = data.groupby('index').count()['geometry'].values
    return data_count

#----------------------
# Get Attribute Function
#----------------------
def get_attributes(parameters,
                   geodata_community,
                   geodata_tract,
                   geodata_block,
                   geodata_extreme):
    
    """
    
    This function is used to process each year's taxi data and
    get node attributes in the graph.
    
    Arg:
        - parameters: configuration of year information
        
    """
    
    ## load parameters 
    root = parameters['taxi-root']
    geodata_community = geopandas.GeoDataFrame(geodata_community, geometry='geometry')
    geodata_tract = geopandas.GeoDataFrame(geodata_tract, geometry='geometry')
    geodata_block = geopandas.GeoDataFrame(geodata_block, geometry='geometry')
    geodata_extreme = geopandas.GeoDataFrame(geodata_extreme, geometry='geometry')
    
    ## load all data
    X_community = []
    X_tract = []
    X_block = []
    X_extreme = []
    
    print(f"   -- Load data...")
    data = pd.read_csv(root)
    data = data[['Pickup Centroid Latitude', 'Pickup Centroid Longitude', 'Trip Start Timestamp']]
    data.columns = ['lat', 'long', 'pickuptime']
    data = data.dropna().reset_index(drop=True)
    whole_data = geopandas.GeoDataFrame(data, geometry=geopandas.points_from_xy(data.long, data.lat))       
    
    ## extract date & time
    dates = [record[:5] for record in whole_data.pickuptime]
    times = np.array([record[11:13]+record[-2:] for record in whole_data.pickuptime])
    times[times=='12AM'] = '00AM'
    times[times=='12PM'] = '12'
    times = np.array(list(map(change_time_format, times)))
    whole_data['date'] = dates
    whole_data['time'] = times
    
    ## unique dates & unique times & segmentations
    UNIQUE_DATES = np.unique(dates)
    UNIQUE_TIME = np.unique(times)
    
    ## attribute data
    print(f"   -- Prepare node attributes...")
    
    for uni_date in tqdm(UNIQUE_DATES):
        for uni_time in UNIQUE_TIME:

            ## subset data
            query = f"date == '{uni_date}' & time == '{uni_time}'"
            hourly_data = whole_data.query(query)
            
            community_count = count(geodata_community, hourly_data)
            tract_count = count(geodata_tract, hourly_data)
            block_count = count(geodata_block, hourly_data)
            extreme_count = count(geodata_extreme, hourly_data)

            X_community.append(community_count)
            X_tract.append(tract_count)
            X_block.append(block_count)
            X_extreme.append(extreme_count)
    
    ## stack data
    X_community = np.stack(X_community)
    X_tract = np.stack(X_tract)
    X_block = np.stack(X_block)
    X_extreme = np.stack(X_extreme)
        
    np.save(f'D:/disaggregation-data/chicago-taxi/attributes/community.npy', X_community)
    np.save(f'D:/disaggregation-data/chicago-taxi/attributes/tract.npy', X_tract)
    np.save(f'D:/disaggregation-data/chicago-taxi/attributes/block.npy', X_block)
    np.save(f'D:/disaggregation-data/chicago-taxi/attributes/extreme.npy', X_extreme)
        
    print(f"Done!")