'''
DBSCAN FOR ME TO UNDERSTAND 
DBSCAN, or Density-Based Spatial Clustering of Applications with Noise, is a data clustering algorithm that groups together points that are close to each other in space. It is a density-based algorithm, which means that it groups together points based on their density in the data space.

DBSCAN works by first defining two parameters: epsilon (eps) and minimum points (minPts). Epsilon is the maximum distance between two points for them to be considered neighbors. Minimum points is the minimum number of points that must be within epsilon of a point for it to be considered a core point.

Once these parameters have been defined, DBSCAN works as follows:

1. It starts by finding all of the core points in the data set. A core point is a point that has at least minPts neighbors within epsilon of it.
2. Once all of the core points have been found, DBSCAN groups together all of the points that are within epsilon of a core point. These points are said to be reachable from the core point.
3. DBSCAN continues to group together points in this way until all of the core points have been visited.
4. Any points that are not reachable from a core point are labeled as noise.

DBSCAN is a powerful clustering algorithm that is able to identify clusters of arbitrary shape and size. It is also able to identify clusters in noisy data sets.

Here is an easy way to understand how DBSCAN works:

Imagine that you have a group of people standing in a room. Each person represents a data point. You can define epsilon as the distance that two people must be from each other in order to be considered neighbors. You can also define minimum points as the minimum number of people that must be within epsilon of a person for them to be considered a core point.

DBSCAN would then work as follows:

1. It would start by finding all of the core points in the room. A core point is a person that has at least minPts neighbors within epsilon of them.
2. Once all of the core points have been found, DBSCAN would group together all of the people that are within epsilon of a core point. These people are said to be reachable from the core point.
3. DBSCAN would continue to group together people in this way until all of the core points have been visited.
4. Any people that are not reachable from a core point would be labeled as noise.

The resulting groups of people would be the clusters that DBSCAN has identified.

DBSCAN is a powerful clustering algorithm that can be used to identify clusters of data in a variety of applications, such as image processing, natural language processing, and machine learning.

'''



import numpy as np
from kmeans import pairwise_dist
class DBSCAN(object):
    def __init__(self, eps, minPts, dataset):
        self.eps = eps
        self.minPts = minPts
        self.dataset = dataset
    def fit(self):      #DOING
        """Fits DBSCAN to dataset and hyperparameters defined in init().
        Args:
            None
        Return:
            cluster_idx: (N, ) int numpy array of assignment of clusters for each point in dataset
        Hint: Using sets for visitedIndices may be helpful here.
        Iterate through the dataset sequentially and keep track of your points' cluster assignments.
        If a point is unvisited or is a noise point (has fewer than the minimum number of neighbor points), then its cluster assignment should be -1.
        Set the first cluster as C = 0
        """
        N = self.dataset.shape[0]
        visitedIndices = set() #given to be useful 
        C = 0  #given 
        cluster_idx = np.full(N, -1)
        for x in range(N): 
            if x not in visitedIndices:
                neighborIndices = self.regionQuery(x) #region
                if len(neighborIndices) >= self.minPts:
                    self.expandCluster(x,neighborIndices,C,cluster_idx,visitedIndices)
                    C+=1

        return cluster_idx
        
        
        #raise NotImplementedError

    def expandCluster(self, index, neighborIndices, C, cluster_idx, visitedIndices):  ##doing tomorrow 
        """Expands cluster C using the point P, its neighbors, and any points density-reachable to P and updates indices visited, cluster assignments accordingly
           HINT: regionQuery could be used in your implementation
        Args:
            index: index of point P in dataset (self.dataset)
            neighborIndices: (N, ) int numpy array, indices of all points witin P's eps-neighborhood
            C: current cluster as an int
            cluster_idx: (N, ) int numpy array of current assignment of clusters for each point in dataset
            visitedIndices: set of indices in dataset visited so far
        Return:
            None
        Hints:  
            1. np.concatenate(), and np.sort() may be helpful here. A while loop may be better than a for loop.
            2. Use, np.unique(), np.take() to ensure that you don't re-explore the same Indices. This way we avoid redundancy.
        """
        cluster_idx[index] = C 
        visitedIndices.add(index)
        while len(neighborIndices) > 0:
            Nei_index = neighborIndices[0]

            if Nei_index not in visitedIndices:
                new_neighbors = self.regionQuery(Nei_index)
                visitedIndices.add(Nei_index)
                cluster_idx[Nei_index] = C
                if len(new_neighbors) >= self.minPts:
                    neighborIndices = np.append(neighborIndices, new_neighbors)
            neighborIndices[:-1] = neighborIndices[1:]
            neighborIndices = neighborIndices[:-1]

        
        
        #raise NotImplementedError

    def regionQuery(self, pointIndex):
        """Returns all points within P's eps-neighborhood (including P)

        Args:
            pointIndex: index of point P in dataset (self.dataset)
        Return:
            indices: (I, ) int numpy array containing the indices of all points within P's eps-neighborhood
        Hint: pairwise_dist (implemented above) and np.argwhere may be helpful here
        """
        N = self.dataset.shape[0]
        dims = self.dataset[pointIndex].shape[0]
        data = self.dataset[pointIndex].reshape(1,dims)
        
        dist = pairwise_dist(self.dataset,data).reshape(N)

        indices = np.argwhere(dist <= self.eps).flatten()

        return indices
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        #raise NotImplementedError
        