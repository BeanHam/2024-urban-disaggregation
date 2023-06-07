import numpy as np
import pandas as pd
import json
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
from shapely import wkt
from utils import *

root = 'D:/disaggregation-data/bikeshare/'

'''
Generate images for domain adaptation task:

    - vmax values are set to the maximum values of the data at the low resolution.
    
    - E.g. puma images: vmax=np.max(puma_att)
           nta images: vmax=np.max(puma_att)
'''

def load_geodata_attributes(name):
    
    ## geodata
    geodata = pd.read_csv(root+f'geodata/{name}.csv')
    geodata['geometry'] = geodata['geometry'].apply(wkt.loads)
    geodata = geopandas.GeoDataFrame(geodata, crs='epsg:4326')
    
    ## attributes
    att = np.load(root+f'attributes/{name}.npy')
    
    return geodata, att

def remove_boundaries_lines(boundaries):
    
    ## iterate each boundary
    for bound1 in tqdm(boundaries):
        coords1 = np.stack(np.where(bound1<255.)).T
        coords1 = list(zip(coords1[:, 0], coords1[:, 1]))
        
        ## iterate neighbor boundary
        for bound2 in boundaries:
            coords2 = np.stack(np.where(bound2<255.)).T
            coords2 = list(zip(coords2[:, 0], coords2[:, 1]))
            
            ## self
            if coords1 == coords2:
                continue
            
            ## intersection
            intersection = list(set(coords1).intersection(set(coords2)))
            if len(intersection) == 0:
                continue
            
            for inter in intersection:
                if len(coords1)>1:
                    bound1[inter] = 255.
                    coords1.remove(inter)
                else:
                    bound2[inter] = 255.
    return boundaries
    
def main():
    
    #--------------------
    # load geodatas
    #--------------------
    geodata_puma, att_puma = load_geodata_attributes('puma')
    geodata_nta, att_nta = load_geodata_attributes('nta')
    geodata_tract, att_tract = load_geodata_attributes('tract')
    geodata_block, att_block = load_geodata_attributes('block')
    geodata_extreme, att_extreme = load_geodata_attributes('extreme')
    
    #--------------------
    # parameters
    #--------------------
    geodatas = [geodata_puma, geodata_nta, geodata_tract, geodata_block, geodata_extreme]
    atts = [att_puma, att_nta, att_tract, att_block, att_extreme]
    names = ['puma', 'nta', 'tract', 'block', 'extreme']
    #img_size = (256, 512)
    #fig_size = (2.56, 5.12)
    img_size = (128, 256)
    fig_size = (1.28, 2.56)

    #--------------------
    # iterate different resolutions
    #--------------------
    for name in names:
        
        print(f'Processing {name} data...')
        
        #--------------------
        # geodata
        #--------------------
        geodata = pd.read_csv(root+f'geodata/{name}.csv')
        geodata['geometry'] = geodata['geometry'].apply(wkt.loads)
        geodata = geopandas.GeoDataFrame(geodata, crs='epsg:4326')
        att = np.load(root+f'attributes/{name}.npy')
        
        #--------------------
        # boundaries
        #--------------------
        boundaries = []
        for i in tqdm(range(len(geodata))):
            index_array = np.zeros(len(geodata))
            index_array[i] = 1.0
            geodata['counts'] = index_array
            f, ax = plt.subplots(1, figsize=fig_size)
            ax = geodata.plot(
                ax=ax,
                column='counts',
                cmap="binary")
            ax.set_axis_off()
            ax.margins(0)
            ax.apply_aspect()
            ax.figure.savefig('img.png', dpi=600, bbox_inches='tight')
            plt.close()
            
            #--------------------
            # grayscale conversion
            #--------------------
            boundary = Image.open('img.png').convert('L').resize(img_size)
            boundaries.append(np.array(boundary))
        boundaries = np.stack(boundaries)
        
        #--------------------
        # Remove Boundaries
        #--------------------
        boundaries = remove_boundaries_lines(boundaries)
        
        #--------------------
        # Attribute Images
        #--------------------
        imgs = []
        for i in tqdm(range(len(att))):
            
            ## generate images
            geodata['counts'] = att[i]
            f, ax = plt.subplots(1, figsize=fig_size)
            ax = geodata.plot(
                ax=ax,
                column='counts',
                cmap="binary",
                vmin=0)
            ax.set_axis_off()
            ax.margins(0)
            ax.apply_aspect()
            ax.figure.savefig('img.png', dpi=600, bbox_inches='tight')
            plt.close()
            
            #--------------------
            # grayscale conversion
            #--------------------
            img = Image.open('img.png').convert('L').resize(img_size)
            imgs.append(np.array(img).astype('float'))
        imgs = np.stack(imgs)
        
        #--------------------
        # Pixel to Counts
        #--------------------
        all_bound = np.sum(boundaries<255., axis=0)
        count_imgs = []
        for i in tqdm(range(len(imgs))):
            single_img = imgs[i]
            single_att = att[i]
            for j in range(len(boundaries)):
                x,y=np.where(boundaries[j]<255.)
                single_img[x,y] = single_att[j]/len(x) ## count/pixel
            single_img[np.where(all_bound==0)] = 0
            count_imgs.append(single_img)
        count_imgs = np.stack(count_imgs)
                
        ## save image array
        np.save(root+f'img-data/{name}_boundaries.npy', boundaries)
        np.save(root+f'img-data/{name}_imgs_all.npy', count_imgs)
        
if __name__ == "__main__":
    main()
    
    
    
    
    
    