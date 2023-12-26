import numpy as np

def create_nl_feature(X):
    '''
    TODO - Create additional features and add it to the dataset
    
    returns:
        X_new - (N, d + num_new_features) array with 
                additional features added to X such that it
                can classify the points in the dataset.
    '''
    d_new = np.exp(-np.sqrt(2)*( X[:,0]**2 + X[:,1]**2))

    return np.append(X, d_new.reshape(-1,1), axis=1)
    # raise NotImplementedError
