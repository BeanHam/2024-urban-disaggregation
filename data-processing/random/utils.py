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


def split_polygon(polygon):
    
    """
    This function is used to split the given polygon into TWO pieces.
    
    Arg:
        - polygon: given polygon to be split
        
    Output:
    
        - two polygons
    """
    
    ## get all boundary points
    boundary_points = list(polygon.boundary.coords)
    
    ## polygon area
    area = polygon.area
    
    ## all combinations of pairs of boundary points
    combs = list(itertools.product(boundary_points, boundary_points))
    
    ## iterations of all combinations
    for start, end in combs:
        
        ## segment line
        line = LineString([start, end])
        merged = linemerge([polygon.boundary, line])
        borders = unary_union(merged)
        polygons = polygonize(borders)
        
        ## split
        multi_polygons = [p for p in polygons]
    
        ## sanity check:
            ## if it's more than 2 polygons or the areas is too 
            ## small (<1/3 * area), check next combination
            ## otherwise, return the split
        length_check = len(multi_polygons)==2
        area_check = np.all([single_polygon.area>(area/3) for single_polygon in multi_polygons])
        
        if length_check & area_check: break
    
    return multi_polygons


def change_time_format(time):
    
    """
    This function is used to change time format.
    """
    if 'PM' in time: return str(int(time[:2])+12)
    else: return time[:2]
    
    
def get_geodata(parameters):
    
    
    """
    This function is used to process geodata for taxi zones:
        
        Given taxi zones of interest, we extract the polygons at the low resolution. 
        Then we split each into two polygons to create a super resolution version.
        
    Arg:
        - parameters: configuration of root and zones
    
    Output:
        - geodata, low resolution
        - geodata, super resolution
        
    """
    
    ### load parameters
    root = parameters['geodata_root']
    zones = parameters['zones']
    
    ### load zone data
    with open(root) as f: 
        gj = geojson.load(f)
        
    ### geodata low resolution
    geodata_low_resolution = []
    for data in gj['features']:
        ID = int(data['properties']['objectid'])
        coords = Polygon(data['geometry']['coordinates'][0][0])
        geodata_low_resolution.append([ID, coords])
    geodata_low_resolution = pd.DataFrame(geodata_low_resolution, columns=['ID', 'polygon'])
    geodata_low_resolution = geodata_low_resolution[np.isin(geodata_low_resolution.ID, zones)].reset_index(drop=True)
    
    ### geodata super resolution
    geodata_super_resolution = []
    for i in range(len(geodata_low_resolution)):
        polygon = geodata_low_resolution.polygon.values[i]
        ID = geodata_low_resolution.ID.values[i]
        multi_polygons = split_polygon(polygon)
        new_id = 1
        for single_polygon in multi_polygons:
            geodata_super_resolution.append([ID, new_id, single_polygon])
            new_id += 1
    geodata_super_resolution = pd.DataFrame(geodata_super_resolution, columns=['parental_ID', 'sub_ID', 'polygon'])
    
    ### adjacency matrix low resolution
    adjacency_low_resolution = np.zeros((geodata_low_resolution.shape[0], geodata_low_resolution.shape[0]))
    for i in range(len(geodata_low_resolution)):
        for j in range(len(geodata_low_resolution)):
            adjacency_low_resolution[i,j] = geodata_low_resolution.polygon[i].intersects(geodata_low_resolution.polygon[j])
    
    #### adjacency matrix super resolution
    adjacency_super_resolution = np.zeros((geodata_super_resolution.shape[0], geodata_super_resolution.shape[0]))
    for i in range(len(geodata_super_resolution)):
        for j in range(len(geodata_super_resolution)):
            adjacency_super_resolution[i,j] = geodata_super_resolution.polygon[i].intersects(geodata_super_resolution.polygon[j])
    
    return geodata_low_resolution, \
           geodata_super_resolution, \
           adjacency_low_resolution, \
           adjacency_super_resolution
    
    
def get_attribute_data(parameters,
                       geodata_low_resolution, 
                       geodata_super_resolution):
    
    """
    
    This function is used to process each year's taxi data and
    get node attributes in the graph.
    
    Arg:
        - parameters: configuration of year information
        - geodata_low_resolution: polygons of taxi zones at low resolution
        - geodata_super_resolution: polygons of taxi zones at super resolution
        
    """
    
    ## load parameters 
    year = parameters['year']
    root = parameters['root']
    column_names = parameters['column_names']
    variables = column_names[year]
    uni_columns = parameters['uni_columns']
    lat_upper = parameters['lat_upper']
    lat_bottom = parameters['lat_bottom']
    long_right = parameters['long_right']
    long_left = parameters['long_left']
    
    ###################### load all data ############################
    print(f"   --Load {year} data...")
    dirt = root + year + '/'
    files = os.listdir(dirt)
    
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
        data['points'] = [Point(data.long.values[i], data.lat.values[i]) for i in range(len(data))]
        whole_data.append(data.values)
    whole_data = pd.DataFrame(np.concatenate(whole_data), columns=uni_columns+['points'])
    
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
    print(f"   --Prepare {year} node attributes...")
    node_attributes_low_resolution = []
    node_attributes_super_resolution = []
    
    for uni_date in UNIQUE_DATES:
        for uni_time in UNIQUE_TIME:    
            ## subset data
            query = f"date == '{uni_date}' & time == '{uni_time}'"
            hourly_data = whole_data.query(query)
            
            ## count taxi rides within each polygon - low resolutioin
            count = []
            for poly in geodata_low_resolution.polygon:
                count.append(sum([poly.contains(point) for point in hourly_data.points]))
            node_attributes_low_resolution.append(count)
            
            ## count taxi rides within each polygon - low resolutioin
            count = []
            for poly in geodata_super_resolution.polygon:
                count.append(sum([poly.contains(point) for point in hourly_data.points]))
            node_attributes_super_resolution.append(count)
            
    node_attributes_low_resolution = np.stack(node_attributes_low_resolution)
    node_attributes_super_resolution = np.stack(node_attributes_super_resolution)
    
    return node_attributes_low_resolution, node_attributes_super_resolution
    