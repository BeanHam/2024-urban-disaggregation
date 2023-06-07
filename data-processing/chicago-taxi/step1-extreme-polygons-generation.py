import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from shapely import wkt
from pointpats import random
from sklearn.cluster import KMeans
from geovoronoi import voronoi_regions_from_coords, points_to_coords
import warnings
warnings.filterwarnings("ignore")

def main():
    
    #--------------------
    # block geodata
    #--------------------
    geodata = pd.read_csv(f'geodata/block.csv')
    geodata['geometry'] = geodata['geometry'].apply(wkt.loads)
    geodata = gpd.GeoDataFrame(geodata, crs='epsg:4326')
    
    #--------------------
    # parameters
    #--------------------
    seed=100
    size=1000
    n_clusters=3
    np.random.seed(seed)
    extreme_polygons = []
    
    #--------------------
    # split
    #--------------------
    for geom in tqdm(geodata.geometry):
        geom = geom.buffer(0)
        try:
            coords = random.poisson(geom, size=size)
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(coords)
            centers = kmeans.cluster_centers_
            sub_polys, _ = voronoi_regions_from_coords(centers, geom)
            for key in sub_polys:
                sub_poly = sub_polys[key]
                if sub_poly.geom_type == 'MultiPolygon':
                    for poly in list(sub_poly.geoms):
                        extreme_polygons.append(poly)
                else:
                    extreme_polygons.append(sub_poly)
        except:
            extreme_polygons.append(geom)
    
    #--------------------
    # save model
    #--------------------
    extrem_data = pd.DataFrame(extreme_polygons, columns=['the_geom'])
    extrem_data.to_csv('raw-data/nyc_extreme_2010.csv', index=False)
    
if __name__ == "__main__":
    main()
    