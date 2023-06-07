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
    gdf = gdf[np.isin(gdf.OBJECTID, zones)]
    
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
    
    ## generate geo adjacency matrix
    geo_A = np.zeros((geodata.shape[0], geodata.shape[0]), dtype='int8')
    for i in tqdm(range(len(geodata))):
        for j in range(len(geodata)):
            geo_A[i,j] = geodata.geometry[i].intersects(geodata.geometry[j])

    ## generate adjacency matrix
    distance_A = np.zeros((geodata.shape[0], geodata.shape[0]), dtype='float')
    for i in tqdm(range(len(geodata))):
        for j in range(len(geodata)):
            x1,y1= geodata.geometry[i].centroid.xy
            x2,y2= geodata.geometry[j].centroid.xy
            distance_A[i,j] = hs.haversine((x1[0], y1[0]), (x2[0], y2[0]))
    distance_A = 1/distance_A
    distance_A[np.isinf(distance_A)]=0
            
    ## generate border matrix
    border_A = np.zeros((geodata.shape[0], geodata.shape[0]), dtype='float')
    for i in tqdm(range(len(geodata))):
        for j in range(len(geodata)):
            line = geodata.geometry[i].intersection(geodata.geometry[j])
            border_A[i,j] = line.length
    np.fill_diagonal(border_A, 0)
    
    ## save  
    geodata.to_csv(f'D:/disaggregation-data/bikeshare/geodata/{data_name}.csv',header=True)
    np.save(f'D:/disaggregation-data/bikeshare/adjacencies/{data_name}-geo.npy', geo_A)
    np.save(f'D:/disaggregation-data/bikeshare/adjacencies/{data_name}-distance.npy', distance_A)
    np.save(f'D:/disaggregation-data/bikeshare/adjacencies/{data_name}-border.npy', border_A)
    
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

    np.save(f'D:/disaggregation-data/bikeshare/linkages/{linkage_name}.npy', linkage)
    
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
                   geodata_puma, 
                   geodata_nta,
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
    year = parameters['year']
    root = parameters['root']
    column_names = parameters['column_names']
    uni_columns = parameters['uni_columns']
    lat_upper = parameters['lat_upper']
    lat_bottom = parameters['lat_bottom']
    long_right = parameters['long_right']
    long_left = parameters['long_left']
    
    ## change geodata from pd.Dataframe to geo.Dataframe
    geodata_puma = geopandas.GeoDataFrame(geodata_puma, geometry='geometry')
    geodata_nta = geopandas.GeoDataFrame(geodata_nta, geometry='geometry')
    geodata_tract = geopandas.GeoDataFrame(geodata_tract, geometry='geometry')
    geodata_block = geopandas.GeoDataFrame(geodata_block, geometry='geometry')
    geodata_extreme = geopandas.GeoDataFrame(geodata_extreme, geometry='geometry')
    
    ## load all data
    X_puma = []
    X_nta = []
    X_tract = []
    X_block = []
    X_extreme = []
    
    print(f"   -- Load {year} data...")
    dirt = root + str(year) + '/'
    files = os.listdir(dirt)
    
    ## get variable names
    variables = [
        ['starttime', 'start station latitude', 'start station longitude'],
        ['started_at', 'start_lat', 'start_lng']
    ]
    uni_columns = ['time', 'lat', 'long']
    
    ## iterate each chunk file
    whole_data = []
    for i in range(len(files)):
        file = files[i]
        data = pd.read_csv(dirt + file)
        try: data = data[variables[0]]
        except: data = data[variables[1]]
        data.columns = uni_columns
        data = data.loc[(data.lat<= lat_upper) &\
                        (data.lat>= lat_bottom) &\
                        (data.long<= long_right) &\
                        (data.long>= long_left)]
        data = geopandas.GeoDataFrame(data, geometry=geopandas.points_from_xy(data.long, data.lat))
        whole_data.append(data.values)
    whole_data = pd.DataFrame(np.concatenate(whole_data), columns=uni_columns+['geometry'])
    whole_data = geopandas.GeoDataFrame(whole_data, geometry='geometry')
    
    ## unique dates & hours
    dates = [record.split()[0] for record in whole_data.time]
    hours = [record.split()[1].split(':')[0] for record in whole_data.time]
    whole_data['date'] = dates
    whole_data['time'] = hours
    UNIQUE_DATES = np.unique(dates)
    UNIQUE_TIME = np.unique(hours)
    
    ## iterate unique date & time
    for uni_date in tqdm(UNIQUE_DATES):
        for uni_time in UNIQUE_TIME:
            
            ## subset data
            query = f"date == '{uni_date}' & time == '{uni_time}'"
            hourly_data = whole_data.query(query)
            
            if len(hourly_data)>0:
                
                puma_count = count(geodata_puma, hourly_data)
                nta_count = count(geodata_nta, hourly_data)
                tract_count = count(geodata_tract, hourly_data)
                block_count = count(geodata_block, hourly_data)
                extreme_count = count(geodata_extreme, hourly_data)
    
                X_puma.append(puma_count)
                X_nta.append(nta_count)
                X_tract.append(tract_count)
                X_block.append(block_count)
                X_extreme.append(extreme_count)
                
            else:
                puma_count = np.zeros(len(geodata_puma))
                nta_count = np.zeros(len(geodata_nta))
                tract_count = np.zeros(len(geodata_tract))
                block_count = np.zeros(len(geodata_block))
                extreme_count = np.zeros(len(geodata_extreme))
                
                X_puma.append(puma_count)
                X_nta.append(nta_count)
                X_tract.append(tract_count)
                X_block.append(block_count)
                X_extreme.append(extreme_count)
    
    ## aggregate
    X_puma = np.stack(X_puma)
    X_nta = np.stack(X_nta)
    X_tract = np.stack(X_tract)
    X_block = np.stack(X_block)
    X_extreme = np.stack(X_extreme)
    
    ## half year -- to be consistent with taxi data
    X_puma = X_puma[:int(len(X_puma)/2)]
    X_nta = X_nta[:int(len(X_nta)/2)]
    X_tract = X_tract[:int(len(X_tract)/2)]
    X_block = X_block[:int(len(X_block)/2)]
    X_extreme = X_extreme[:int(len(X_extreme)/2)]
    
    ## save data
    np.save(f'D:/disaggregation-data/bikeshare/attributes/puma.npy', X_puma)
    np.save(f'D:/disaggregation-data/bikeshare/attributes/nta.npy', X_nta)
    np.save(f'D:/disaggregation-data/bikeshare/attributes/tract.npy', X_tract)
    np.save(f'D:/disaggregation-data/bikeshare/attributes/block.npy', X_block)
    np.save(f'D:/disaggregation-data/bikeshare/attributes/extreme.npy', X_extreme)
    
    print(f"Done!")
    