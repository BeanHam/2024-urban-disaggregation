import os
import json
import time
import shapely
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
def get_boundary(root, parameters):
    
    """
    This function is used to generate the Manhanttan boundary
    
    """
    
    ## load data & zones
    data = pd.read_csv(root)
    zones = parameters['zones']
    
    ## make the data -> geodata
    data = data.rename(columns={'the_geom':'geometry'})
    data['geometry'] = data['geometry'].apply(wkt.loads)
    gdf = geopandas.GeoDataFrame(data, crs='epsg:4326')
    #gdf = gdf[np.isin(gdf.DIST_NUM, zones)]
    gdf = gdf[np.isin(gdf.LocationID, zones)]
    
    ## generate boundary
    new_polygons = []
    for geom in gdf.geometry:
        multi_polygons = list(geom.geoms)
        areas = [poly.area for poly in multi_polygons]
        new_polygons.append(multi_polygons[np.where(areas == np.max(areas))[0][0]])
    gdf.geometry = new_polygons
    polygons = [i for i in gdf.geometry.values]
    boundary = unary_union(polygons)
    
    return boundary
    
#----------------------
# Get Extreme Data
#----------------------
def get_extreme_data(geodata_block, data_name, n_splits=3):
    
    """
    Split each city block into 3 extreme blocks
    """
        
    # generate extreme polygons
    extreme_polygons = []
    for geom in tqdm(geodata_block.geometry):
        try:
            geom = geom.buffer(0)
            centroid = list(geom.centroid.coords)[0]
            splits = np.array_split(list(geom.boundary.coords)[1:], n_splits)
            start = 0
            index = True
            while index:
                line_split_collection = [LineString([centroid, p[start]]) for p in splits]
                line_split_collection.append(geom.boundary)
                merged_lines = shapely.ops.linemerge(line_split_collection)
                border_lines = shapely.ops.unary_union(merged_lines)
                polygons = shapely.ops.polygonize(border_lines)            
                if len(polygons) == n_splits:
                    for poly in polygons:
                        extreme_polygons.append(poly)
                    index=False
                else:
                    start += 1            
        except:
            extreme_polygons.append(geom)
       
    extrem_data = pd.DataFrame(extreme_polygons, columns=['the_geom'])
    extrem_data.to_csv(f'D:/disaggregation-data/{data_name}/raw-data/extreme_2010.csv', index=False)       
    
#----------------------
# Get GeoData Function
#----------------------
def get_geodata(root, 
                boundary, 
                data_name,
                resolution_name):
    
    if os.path.isfile(f'D:/disaggregation-data/{data_name}/geodata/{resolution_name}.csv'):
        data = pd.read_csv(f'D:/disaggregation-data/{data_name}/geodata/{resolution_name}.csv')
        data = data.rename(columns={'the_geom':'geometry'})
        data['geometry'] = data['geometry'].apply(wkt.loads)
        geodata = geopandas.GeoDataFrame(data, crs='epsg:4326')    
    else:
        ## load data & make geodata
        data = pd.read_csv(root)
        data = data.rename(columns={'the_geom':'geometry'})
        data['geometry'] = data['geometry'].apply(wkt.loads)
        gdf = geopandas.GeoDataFrame(data, crs='epsg:4326')
        
        ## subset regions within boundary
        geodata = []
        for i in tqdm(range(len(gdf))):
            geom = gdf.geometry[i]
            if geom.is_valid == False: geom = geom.buffer(0)            
            if geom.type == 'Polygon': multi_polygons = [geom]
            else: multi_polygons = list(geom.geoms)            
            for polygon in multi_polygons:
                if boundary.contains(polygon.centroid):
                    area = polygon.area*10**11
                    geodata.append([polygon, area])
        geodata = pd.DataFrame(geodata, columns=['geometry', 'shape_area'])
        
        ## generate geo adjacency matrix
        geo_A = np.zeros((geodata.shape[0], geodata.shape[0]), dtype='int8')
        for i in tqdm(range(len(geodata))):
            for j in range(len(geodata)):
                geo_A[i,j] = geodata.geometry[i].intersects(geodata.geometry[j])
        
        ## save  
        geodata.to_csv(f'D:/disaggregation-data/{data_name}/geodata/{resolution_name}.csv',header=True,index=False)
        np.save(f'D:/disaggregation-data/{data_name}/adjacencies/{resolution_name}.npy', geo_A)    
    
    return geodata

#----------------------
# Get Linkage Function
#----------------------
def get_linkage(low_res_geodata, 
                high_res_geodata, 
                data_name,
                linkage_name):
    
    """
    
    This function is used to get the linkage between low resolution graph to
    super resolution graph using geodata.
    
    
    """
    
    if os.path.isfile(f'D:/disaggregation-data/{data_name}/linkages/{linkage_name}.npy'):
        linkage = np.load(f'D:/disaggregation-data/{data_name}/linkages/{linkage_name}.npy')
    else:
        linkage = np.zeros((low_res_geodata.shape[0], high_res_geodata.shape[0]), dtype='int8')
        for i in tqdm(range(len(low_res_geodata))):
            for j in range(len(high_res_geodata)):
                linkage[i,j] = low_res_geodata.geometry[i].contains(high_res_geodata.geometry[j].centroid)    
        np.save(f'D:/disaggregation-data/{data_name}/linkages/{linkage_name}.npy', linkage)
    
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
def get_attributes_taxi(parameters,
                        geodata_puma, 
                        geodata_nta,
                        geodata_tract,
                        geodata_block):
    
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
    
    ## load all data
    X_puma = []
    X_nta = []
    X_tract = []
    X_block = []
    
    print(f"   -- Load {year} data...")
    dirt = root + year + '/'
    files = os.listdir(dirt)
    variables = column_names[year]
    
    whole_data = []
    for file in files:
        
        data = pd.read_csv(dirt + file)
        data = data[variables]
        data.columns = uni_columns
        
        ## subset data to regions of our interest
        ## (for faster processing purpose as well)        
        data = data.loc[(data.lat<= lat_upper) &\
                        (data.lat>= lat_bottom) &\
                        (data.long<= long_right) &\
                        (data.long>= long_left)]
        data = geopandas.GeoDataFrame(data, geometry=geopandas.points_from_xy(data.long, data.lat))
        whole_data.append(data.values)
    whole_data = pd.DataFrame(np.concatenate(whole_data), columns=uni_columns+['geometry'])
    whole_data = geopandas.GeoDataFrame(whole_data, geometry='geometry')
    
    ## extract date & time
    if year != '2016':
        dates = [record[5:10] for record in whole_data.pickup_datetime]
        times = [record[11:13] for record in whole_data.pickup_datetime]
        whole_data['date'] = dates
        whole_data['time'] = times
    else:
        dates = [record[:5] for record in whole_data.pickup_datetime]
        times = [record[11:13]+record[-2:] for record in whole_data.pickup_datetime]
        times = np.array(list(map(change_time_format, times)))
        
        ## 12 -> 12AM; should be changed to 00
        ## 24 -> 12PM; should stay as 12
        times[times == '12'] = '00'
        times[times == '24'] = '12'
        whole_data['date'] = dates
        whole_data['time'] = times
        
    ## unique dates & unique times & segmentations
    UNIQUE_DATES = np.unique(dates)
    UNIQUE_TIME = np.unique(times)
    
    ## attribute data
    print(f"   -- Prepare {year} node attributes...")
    
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
    
                X_puma.append(puma_count)
                X_nta.append(nta_count)
                X_tract.append(tract_count)
                X_block.append(block_count)
                
            else:
                puma_count = np.zeros(len(geodata_puma))
                nta_count = np.zeros(len(geodata_nta))
                tract_count = np.zeros(len(geodata_tract))
                block_count = np.zeros(len(geodata_block))
                
                X_puma.append(puma_count)
                X_nta.append(nta_count)
                X_tract.append(tract_count)
                X_block.append(block_count)
    
    ## stack data
    X_puma = np.stack(X_puma)
    X_nta = np.stack(X_nta)
    X_tract = np.stack(X_tract)
    X_block = np.stack(X_block)
        
    np.save(f'D:/disaggregation-data/taxi/attributes/puma.npy', X_puma)
    np.save(f'D:/disaggregation-data/taxi/attributes/nta.npy', X_nta)
    np.save(f'D:/disaggregation-data/taxi/attributes/tract.npy', X_tract)
    np.save(f'D:/disaggregation-data/taxi/attributes/block.npy', X_block)
        
    print(f"Done!")
    
    
#----------------------
# Get Attribute Function
#----------------------
def get_attributes_bikeshare(parameters,
                             geodata_puma, 
                             geodata_nta,
                             geodata_tract,
                             geodata_block):
    
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
    
    ## load all data
    X_puma = []
    X_nta = []
    X_tract = []
    X_block = []
    
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
    #UNIQUE_DATES = np.array(UNIQUE_DATES)[np.array(UNIQUE_DATES)<='2021-06-30']
    
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
    
                X_puma.append(puma_count)
                X_nta.append(nta_count)
                X_tract.append(tract_count)
                X_block.append(block_count)
                
            else:
                puma_count = np.zeros(len(geodata_puma))
                nta_count = np.zeros(len(geodata_nta))
                tract_count = np.zeros(len(geodata_tract))
                block_count = np.zeros(len(geodata_block))
                
                X_puma.append(puma_count)
                X_nta.append(nta_count)
                X_tract.append(tract_count)
                X_block.append(block_count)
    
    ## aggregate
    X_puma = np.stack(X_puma)
    X_nta = np.stack(X_nta)
    X_tract = np.stack(X_tract)
    X_block = np.stack(X_block)
        
    ## save data
    np.save(f'D:/disaggregation-data/bikeshare/attributes/puma.npy', X_puma)
    np.save(f'D:/disaggregation-data/bikeshare/attributes/nta.npy', X_nta)
    np.save(f'D:/disaggregation-data/bikeshare/attributes/tract.npy', X_tract)
    np.save(f'D:/disaggregation-data/bikeshare/attributes/block.npy', X_block)
    
    print(f"Done!")
    
    
#----------------------
# Get Attribute Function
#----------------------
def get_attributes_911(parameters,
                       geodata_puma, 
                       geodata_nta,
                       geodata_tract,
                       geodata_block):
    
    """
    
    This function is used to process each year's taxi data and
    get node attributes in the graph.
    
    Arg:
        - parameters: configuration of year information
        
    """
    
    ## load parameters 
    root = parameters['root']
    lat_upper = parameters['lat_upper']
    lat_bottom = parameters['lat_bottom']
    long_right = parameters['long_right']
    long_left = parameters['long_left']
    
    ## change geodata from pd.Dataframe to geo.Dataframe
    geodata_puma = geopandas.GeoDataFrame(geodata_puma, geometry='geometry')
    geodata_nta = geopandas.GeoDataFrame(geodata_nta, geometry='geometry')
    geodata_tract = geopandas.GeoDataFrame(geodata_tract, geometry='geometry')
    geodata_block = geopandas.GeoDataFrame(geodata_block, geometry='geometry')
    
    ## load all data
    X_puma = []
    X_nta = []
    X_tract = []
    X_block = []
    
    data = pd.read_csv(root)
    data = data[data.BORO_NM == 'MANHATTAN']
    data['lat'] = data['Latitude']
    data['long'] = data['Longitude']
    data['date'] = data['CREATE_DATE']
    data['time'] = [record[:2]for record in data.INCIDENT_TIME]
    data = data.loc[(data.lat<= lat_upper) &\
                    (data.lat>= lat_bottom) &\
                    (data.long<= long_right) &\
                    (data.long>= long_left)]
    data = geopandas.GeoDataFrame(data, geometry=geopandas.points_from_xy(data.long, data.lat))
    UNIQUE_DATES = np.unique(data['date'])
    UNIQUE_TIME = np.unique(data['time'])
    #UNIQUE_DATES = np.array(UNIQUE_DATES)[np.array(UNIQUE_DATES)<='06/30/2022']
    
    ## iterate unique date & time
    for uni_date in tqdm(UNIQUE_DATES):
        for uni_time in UNIQUE_TIME:
            
            ## subset data
            query = f"date == '{uni_date}' & time == '{uni_time}'"
            hourly_data = data.query(query)
            
            if len(hourly_data)>0:
                
                puma_count = count(geodata_puma, hourly_data)
                nta_count = count(geodata_nta, hourly_data)
                tract_count = count(geodata_tract, hourly_data)
                block_count = count(geodata_block, hourly_data)
    
                X_puma.append(puma_count)
                X_nta.append(nta_count)
                X_tract.append(tract_count)
                X_block.append(block_count)
                
            else:
                puma_count = np.zeros(len(geodata_puma))
                nta_count = np.zeros(len(geodata_nta))
                tract_count = np.zeros(len(geodata_tract))
                block_count = np.zeros(len(geodata_block))
                
                X_puma.append(puma_count)
                X_nta.append(nta_count)
                X_tract.append(tract_count)
                X_block.append(block_count)
    
    ## aggregate
    X_puma = np.stack(X_puma)
    X_nta = np.stack(X_nta)
    X_tract = np.stack(X_tract)
    X_block = np.stack(X_block)
        
    ## save data
    np.save(f'D:/disaggregation-data/911/attributes/puma.npy', X_puma)
    np.save(f'D:/disaggregation-data/911/attributes/nta.npy', X_nta)
    np.save(f'D:/disaggregation-data/911/attributes/tract.npy', X_tract)
    np.save(f'D:/disaggregation-data/911/attributes/block.npy', X_block)
    
    print(f"Done!")
    
#----------------------
# Get Attribute Function
#----------------------
def get_attributes_chicago(parameters,
                           geodata_community,
                           geodata_tract,
                           geodata_block):
    
    """
    
    This function is used to process each year's taxi data and
    get node attributes in the graph.
    
    Arg:
        - parameters: configuration of year information
        
    """
    
    ## load parameters 
    root = parameters['root']
    geodata_community = geopandas.GeoDataFrame(geodata_community, geometry='geometry')
    geodata_tract = geopandas.GeoDataFrame(geodata_tract, geometry='geometry')
    geodata_block = geopandas.GeoDataFrame(geodata_block, geometry='geometry')
    
    ## load all data
    X_community = []
    X_tract = []
    X_block = []
    
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
            
            if len(hourly_data)>0:
                
                community_count = count(geodata_community, hourly_data)
                tract_count = count(geodata_tract, hourly_data)
                block_count = count(geodata_block, hourly_data)
    
                X_community.append(community_count)
                X_tract.append(tract_count)
                X_block.append(block_count)
                
            else:
                
                community_count = np.zeros(len(geodata_community))
                tract_count = np.zeros(len(geodata_tract))
                block_count = np.zeros(len(geodata_block))
    
                X_community.append(community_count)
                X_tract.append(tract_count)
                X_block.append(block_count)
                    
    ## stack data
    X_community = np.stack(X_community)
    X_tract = np.stack(X_tract)
    X_block = np.stack(X_block)
        
    np.save(f'D:/disaggregation-data/chicago/attributes/community.npy', X_community)
    np.save(f'D:/disaggregation-data/chicago/attributes/tract.npy', X_tract)
    np.save(f'D:/disaggregation-data/chicago/attributes/block.npy', X_block)
        
    print(f"Done!")    