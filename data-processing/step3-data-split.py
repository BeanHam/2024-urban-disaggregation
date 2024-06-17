import numpy as np
import argparse

def main():
    
    #-------------------------
    # arguments
    #-------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='filename of test data')
    args = parser.parse_args()
    data_name = args.data
    
    if data_name == 'taxi': 
        root='D:/disaggregation-data/taxi/'
        names = ['puma', 'nta', 'tract', 'block']
    elif data_name == 'bikeshare': 
        root='D:/disaggregation-data/bikeshare/'
        names = ['puma', 'nta', 'tract', 'block']
    elif data_name == '911': 
        root='D:/disaggregation-data/911/'
        names = ['puma', 'nta', 'tract', 'block']
    else: 
        root='D:/disaggregation-data/chicago/'
        names = ['community', 'tract', 'block']
    
    #-------------------------
    # split
    #-------------------------    
    for name in names:
        
        print(f'Name: {name}')
        
        ## attributes
        data = np.load(root+f"attributes/{name}.npy")
        train, test = data[:-31*24], data[-31*24:]
        train, val = train[:-30*24], train[-30*24:]
        np.save(root+f"attributes/{name}_train.npy", train)
        np.save(root+f"attributes/{name}_val.npy", val)
        np.save(root+f"attributes/{name}_test.npy", test)
        
        ## images
        data = np.load(root+f"img-data/{name}_imgs_all.npy")
        train, test = data[:-31*24], data[-31*24:]
        train, val = train[:-30*24], train[-30*24:]
        np.save(root+f"img-data/{name}_imgs_train.npy", train)
        np.save(root+f"img-data/{name}_imgs_val.npy", val)
        np.save(root+f"img-data/{name}_imgs_test.npy", test)    
        
if __name__ == "__main__":
    main()