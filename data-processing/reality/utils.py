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
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

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
    np.save(f'linkage/{linkage_name}.npy', linkage)
    
    return linkage


def get_attributes(parameters,
                   geodata_puma, 
                   geodata_nta,
                   geodata_taxi,
                   geodata_tract):
    
    """
    
    This function is used to process each year's taxi data and
    get node attributes in the graph.
    
    Arg:
        - parameters: configuration of year information
        
    """
    
    ## load parameters 
    years = parameters['years']
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
    geodata_taxi = geopandas.GeoDataFrame(geodata_taxi, geometry='geometry')
    geodata_tract = geopandas.GeoDataFrame(geodata_tract, geometry='geometry')
    
    ###################### load all data ############################
    X_puma = []
    X_nta = []
    X_taxi = []
    X_tract = []
    
    for year in years:
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
        
        ###################### extract date & time ############################
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
        
        ###################### attribute data low resolution ############################
        print(f"   -- Prepare {year} node attributes...")
        
        iteration = 1
        for uni_date in UNIQUE_DATES:
            for uni_time in UNIQUE_TIME:
                
                if iteration % 500 == 0:
                    print(f'      -- [{iteration}/{len(UNIQUE_DATES)*len(UNIQUE_TIME)}]...')
                
                ## subset data
                query = f"date == '{uni_date}' & time == '{uni_time}'"
                hourly_data = whole_data.query(query)
                
                ## puma
                puma = geopandas.sjoin(geodata_puma, hourly_data)
                puma['index'] = puma.index
                puma_count = np.zeros(len(geodata_puma))
                puma_count[np.unique(puma.index)] = puma.groupby('index').count()['geometry'].values
                X_puma.append(puma_count)
                                
                nta = geopandas.sjoin(geodata_nta, hourly_data)
                nta['index'] = nta.index
                nta_count = np.zeros(len(geodata_nta))
                nta_count[np.unique(nta.index)] = nta.groupby('index').count()['geometry'].values
                X_nta.append(nta_count)
                
                
                taxi = geopandas.sjoin(geodata_taxi, hourly_data)
                taxi['index'] = taxi.index
                taxi_count = np.zeros(len(geodata_taxi))
                taxi_count[np.unique(taxi.index)] = taxi.groupby('index').count()['geometry'].values
                X_taxi.append(taxi_count)
                
                tract = geopandas.sjoin(geodata_tract, hourly_data)
                tract['index'] = tract.index
                tract_count = np.zeros(len(geodata_tract))
                tract_count[np.unique(tract.index)] = tract.groupby('index').count()['geometry'].values
                X_tract.append(tract_count)                
                
                iteration += 1
    
    ## save data
    X_puma = np.stack(X_puma)
    X_nta = np.stack(X_nta)
    X_taxi = np.stack(X_taxi)
    X_tract = np.stack(X_tract)
    
    np.save(f'attributes/puma.npy', X_puma)
    np.save(f'attributes/nta.npy', X_nta)
    np.save(f'attributes/taxi.npy', X_taxi)
    np.save(f'attributes/tract.npy', X_tract)
        
    print(f"Done!")