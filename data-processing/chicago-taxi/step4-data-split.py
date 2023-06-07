import numpy as np
root = 'D:/disaggregation-data/taxi/'

def main():
    
    names = ['puma', 'nta', 'tract', 'block', 'extreme']
    
    for name in names:
        
        print(f'Name: {name}')
        
        ## attributes
        data = np.load(root+f"attributes/{name}.npy")
        train, test = data[:-30*24], data[-30*24:]
        train, val = train[:-31*24], train[-31*24:]
        np.save(root+f"attributes/{name}_train.npy", train)
        np.save(root+f"attributes/{name}_val.npy", val)
        np.save(root+f"attributes/{name}_test.npy", test)
        
        ## images
        data = np.load(root+f"img-data/{name}_imgs_all.npy")
        train, test = data[:-30*24], data[-30*24:]
        train, val = train[:-31*24], train[-31*24:]
        np.save(root+f"img-data/{name}_imgs_train.npy", train)
        np.save(root+f"img-data/{name}_imgs_val.npy", val)
        np.save(root+f"img-data/{name}_imgs_test.npy", test)    
        
if __name__ == "__main__":
    main()