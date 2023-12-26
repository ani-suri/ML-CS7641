'''
File: kmeans.py
Project: Downloads
File Created: Feb 2021
Author: Rohit Das
'''

import numpy as np


class KMeans(object):

    def __init__(self, points, k, init='random', max_iters=10000, rel_tol=1e-5):  # No need to implement
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            init : how to initial the centers
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            rel_tol: convergence criteria with respect to relative change of loss (number between 0 and 1)
        Return:
            none
        """
        self.points = points
        self.K = k
        if init == 'random':
            self.centers = self.init_centers()
        else:
            self.centers = self.kmpp_init()
        self.assignments = None
        self.loss = 0.0
        self.rel_tol = rel_tol
        self.max_iters = max_iters

    def init_centers(self):# [2 pts]  ##DONE 
        """
            Initialize the centers randomly
        Return:
            self.centers : K x D numpy array, the centers.
        Hint: Please initialize centers by randomly sampling points from the dataset in case the autograder fails.
        """
        N = self.points.shape[0]  # N: Number of data points
        # N is size of data, k is number of centroid clusters I want to initialize 
        cluster = np.random.choice(N, self.K, replace=False)
        centers = self.points[cluster]
        
        return centers

    #raise NotImplementedError

    def kmpp_init(self):# [3 pts] #DONE
        """
            Use the intuition that points further away from each other will probably be better initial centers
        Return:
            self.centers : K x D numpy array, the centers.
        """
        '''
        OUTLINE: (for me)
        1) init cent array 
        2) loop- dist, min dist, new dist, update cent
        3) return cent 
        '''
        N = self.points.shape[0] #number of data points of N 
        D = self.points.shape[1]  #number of features of D or the dimensions? 
        # paras set 
        init_centroid = np.random.randint(N) #pick a random int from the N as the initial centroid 
        centers = self.points[init_centroid].reshape(1, D)
        remainting_cent = self.points[init_centroid, :] #initialize centres array with this point 
        
        #iteration for k-1 centers (one point is already chosen as the init_centroid, only k-1 is left)
        '''
        using the pairwise_dist function already implemented 
        x = x[:, np.newaxis, :]
        y = y[np.newaxis, :, :]
        dist = np.sum((x - y)**2, axis=2)
        dist = np.sqrt(dist)
        '''
        
        for x in range(self.K - 1): 
            distance_kmpp = pairwise_dist(self.points, centers)
            min_distance_kmpp = np.min(distance_kmpp, axis=1)
            new_centroid = np.argmax(min_distance_kmpp)
            centers = np.append(centers, self.points[new_centroid, :].reshape(1,D), axis=0)
        return centers
        

       # raise NotImplementedError

    def update_assignment(self):  # [5 pts] #DONE
        """
            Update the membership of each point based on the closest center
        Return:
            self.assignments : numpy array of length N, the cluster assignment for each point
        Hint: You could call pairwise_dist() function
        Hint: In case the np.sqrt() function is giving an error in the pairwise_dist() function, you can use the squared distances directly for comparison. 
        """    
                #dist b/w each point and the cent using pairwise_dist function 
        # Assign data point to closest cent, min of pairise dist 
        # return array where the element in this array is the cluster to which the dp has been assigned 
        dist = pairwise_dist(self.points, self.centers)
        index_cluster = np.argmin(dist, axis=1)
        return  index_cluster
        #raise NotImplementedError

    def update_centers(self):  # [5 pts]  #DONE
        """
            update the cluster centers
        Return:
            self.centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.

        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
        """

        cluster_num=0 
        #K = self.centers[0]
       # D = self.centers[1]
        K, D = self.centers.shape
        #update_assignment same thing 
        # dist = pairwise_dist(self.points, self.centers)
        # self.assignments = np.argmin(dist, axis=1)
        

        up_centers = np.zeros([K, D], dtype=float)
        for cluster_num in range(K):
            
            c = self.points[self.assignments == cluster_num, :]
            centroid = c.mean(axis=0)
            up_centers[cluster_num, :] = centroid
            
            
        self.centers = up_centers

        #self.centers = up_centers  # Update the centers in the object
  
        return up_centers


        #raise NotImplementedError

    def get_loss(self):  # [5 pts] #DONE 
        """
            The loss will be defined as the sum of the squared distances between each point and it's respective center.
        Return:
            self.loss: a single float number, which is the objective function of KMeans.
        """
        #sum of squared distances between each data point and its respective cluster center
        #loss = how far the data points are from the centre 
        #init loss (l) and cluster_val 
        l = 0 
        cluster_val =0 
        for i in self.centers: 
            c = self.points[self.assignments == cluster_val, :]
            l += np.sum(np.square(c - i))
            cluster_val +=1
        return l 
            

         #raise NotImplementedError

    def train(self):    # [10 pts] #DONE  #Autograder test issue, come back to this 
        """
            Train KMeans to cluster the data:
                0. Recall that centers have already been initialized in __init__
                1. Update the cluster assignment for each point
                2. Update the cluster centers based on the new assignments from Step 1
                3. Check to make sure there is no mean without a cluster, 
                   i.e. no cluster center without any points assigned to it.
                   - In the event of a cluster with no points assigned, 
                     pick a random point in the dataset to be the new center and 
                     update your cluster assignment accordingly.
                4. Calculate the loss and check if the model has converged to break the loop early.
                   - The convergence criteria is measured by whether the percentage difference 
                     in loss compared to the previous iteration is less than the given 
                     relative tolerance threshold (self.rel_tol). 
                   - Relative tolerance threshold (self.rel_tol) is a number between 0 and 1.   
                5. Iterate through steps 1 to 4 max_iters times. Avoid infinite looping!
                
        Return:
            self.centers: K x D numpy array, the centers
            self.assignments: Nx1 int numpy array
            self.loss: final loss value of the objective function of KMeans.
        """
        old_l = self.get_loss()
        for iterations in range(self.max_iters): 
            #1) Update cluster assignment 
            self.assignments = self.update_assignment()
            
            #2) update the cluster centers based in the new assignments from step #1 
            self.centers= self.update_centers() 
            
            #3) Checking to make sure there is no mean w/ and empty cluster (no mean left behind)
            for x in range(self.K): 
                cp = self.points[self.assignments == x]
                if len(cp) == 0: #handling for empty cluster 
                    non_clusters = [c for c in range(self.K) if c != x]
                    if non_clusters: 
                        random_cp = np.random.choice(non_clusters)
                        self.centers[x] = self.centers[random_cp]
                        
                else: 
                    np.random.rand(self.centers.shape[1])

    
                    
                    # rand_point = np.random.choice(range(self.points.shape[0]))
                    # self.centers[empty_cluster] = self.points[rand_point]
            
            #4) loss 
            l = self.get_loss()
            if old_l != 0 and (abs(old_l - l) / old_l) < self.rel_tol:
                break 
            
            old_l = l
            
        return self.centers, self.assignments, self.loss



        #raise NotImplementedError


def pairwise_dist(x, y):  # [5 pts] #TO DO 
        np.random.seed(1)
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between
                x[i, :] and y[j, :]
        """
        #Note to self: look at rough.ipynb for working 
        #x_expanded = x[:, np.newaxis, :]
        '''
        1) add extra dims for broadcasting
        2) sum of the sq distances 
        3) sqrt of the sum of the sq distances 
        
        '''
        x = x[:, np.newaxis, :]
        y = y[np.newaxis, :, :]
        dist = np.sum((x - y)**2, axis=2)
        dist = np.sqrt(dist)
        
        return dist 
        #raise NotImplementedError

def rand_statistic(xGroundTruth, xPredicted): # [5 pts]   #DOING 
    """
    Args:
        xPredicted : N x 1 numpy array, N = no. of test samples
        xGroundTruth: N x 1 numpy array, N = no. of test samples
    Return:
        Rand statistic value: final coefficient value as a float
    """
    
    #init all the values 
    TP = 0 
    TN = 0 
    FP = 0 
    FN = 0 
    
    n = len(xGroundTruth)
    
    for x in range(n): 
        for j in range(x +1, n): 
            t = xGroundTruth[x] == xGroundTruth[j]    #truth same 
            p = xPredicted[x] == xPredicted[j]        #predicted same 
            
            if t == p: 
                TP += 1 
            elif not t and not p:
                TN += 1 
            elif t and not p: 
                FN += 1 
            else:                               # both t and p not equal 
                FP += 1 
        
    rand = (TP + TN) / (TP + TN + FP + FN)
    
    return rand 
                
            
            
    
    
    
    
    
    
    
    
    #raise NotImplementedError