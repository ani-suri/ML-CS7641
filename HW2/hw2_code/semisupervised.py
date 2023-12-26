'''
File: semisupervised.py
Project: autograder_test_files
File Created: September 2020
Author: Shalini Chaudhuri (you@you.you)
Updated: September 2022, Arjun Agarwal
'''
import numpy as np
from tqdm import tqdm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

SIGMA_CONST = 1e-6
LOG_CONST = 1e-32

def complete_(data): # [1pts]   #DONE 
    """
    Args:
        data: N x (D+1) numpy array where the last column is the labels 
    # Return:
    #     labeled_complete: n x (D+1) array (n <= N) where values contain both complete features and labels
    # """
    # # data should not have any NaN 
    #find rows where even one col has an nan, this includes the features and the labels'
    rows_with_nan = np.any(np.isnan(data), axis=1)
    rows_without_nan = ~rows_with_nan
    labeled_complete = data[rows_without_nan]
    
    return labeled_complete
    #raise NotImplementedError
    
def incomplete_(data): # [1pts]   #DONE 
    """
    Args:
        data: N x (D+1) numpy array where the last column is the labels
    Return:
        labeled_incomplete: n x (D+1) array (n <= N) where values contain incomplete features but complete labels
    """   
    # incomplete features 
    '''
    data set for incomplete  = data[:, :-1] excludes the last col, takes all rows --> featueres no label 
    '''
    #incomp_features=  data[np.any(np.isnan(data[:, :-1]), axis=1)]
    '''
    data set for complete lavel = data[:, -1] all the rows but the last col 
    '''
    incomp_features=  data[np.any(np.isnan(data[:, :-1]), axis=1)]
    #y_data = incomp_features[:, -1]
    return incomp_features
    #raise NotImplementedError
   


def unlabeled_(data): # [1pts]   #DONE 
    """
    Args:
        data: N x (D+1) numpy array where the last column is the labels   
    Return:
        unlabeled_complete: n x (D+1) array (n <= N) where values contain complete features but incomplete labels
    """
    y = data[:, -1]
    incomp_y = data[np.isnan(y)]
    
    return incomp_y
    #raise NotImplementedError


class CleanData(object):
    def __init__(self): # No need to implement
        pass

    def pairwise_dist(self, x, y): # [0pts] - copy from kmeans
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
            dist: N x M array, where dist[i, j] is the euclidean distance between
            x[i, :] and y[j, :]
        """
        #x = x[:, np.newaxis, :]
        #y = y[np.newaxis, :, :]
        #dist = np.sum((x - y)**2, axis=2)
        dist = np.sum((x - y)**2)

        dist = np.sqrt(dist)
        
        # return dist 

        #raise NotImplementedError
    
    def __call__(self, incomplete_points,  complete_points, K, **kwargs): # [7pts]
        """
        Function to clean or "fill in" NaN values in incomplete data points based on
        the average value for that feature for the K-nearest neighbors in the complete data points. 

        Args:
            incomplete_points: N_incomplete x (D+1) numpy array, the incomplete labeled observations
            complete_points:   N_complete   x (D+1) numpy array, the complete labeled observations
            K: integer, corresponding to the number of nearest neighbors you want to base your calculation on
            kwargs: any other args you want
        Return:
            clean_points: (N_complete + N_incomplete) x (D+1) numpy array, containing both the complete points and recently filled points

        Notes: 
            (1) The first D columns are features, and the last column is the class label
            (2) There may be more than just 2 class labels in the data (e.g. labels could be 0,1,2 or 0,1,2,...,M)
            (3) There will be at most 1 missing feature value in each incomplete data point (e.g. no points will have more than one NaN value)
            (4) You want to find the k-nearest neighbors, from the complete dataset, with the same class labels;
            (5) There may be missing values in any of the features. It might be more convenient to address each feature at a time.
            (6) Do NOT use a for-loop over N_incomplete; you MAY use a for-loop over the M labels and the D features (e.g. omit one feature at a time) 
            (7) You do not need to order the rows of the return array clean_points in any specific manner
        """
        #iterate over features in the data 
        #finds the complete data points with the same class label as the incomplete data points
        # dis b/w complete and incomplete data points 
        # Create a copy of the incomplete data points array, so that we don't modify the original
        # Create a copy of the incomplete data points array, so that we don't modify the original
        clean_points = incomplete_points.copy()
        for feature in range(complete_points.shape[1] - 1):
            for i in range(incomplete_points.shape[0]):
                if np.isnan(clean_points[i, feature]):
                    complete_points_with_same_label = complete_points[complete_points[:, -1] == clean_points[i, -1], :]
                    distances = self.pairwise_dist(clean_points[i, :-1], complete_points_with_same_label[:, :-1])

                    # Find the K-nearest neighbors
                    nearest_neighbors = np.argsort(distances, axis=0)[:K]

                    # Check if the `complete_points_with_same_label` array has a third dimension
                    if complete_points_with_same_label.ndim == 3:
                        # Sum the elements of the array along axis 2
                        clean_points[i, feature] = np.sum(complete_points_with_same_label[nearest_neighbors, feature, :], axis=2).mean()
                    else:
                        # Sum the elements of the array without specifying an axis
                        clean_points[i, feature] = np.sum(complete_points_with_same_label[nearest_neighbors, feature]).mean()

        return clean_points

            
def mean_clean_data(data): # [2pts]
    """
    Args:
        data: N x (D+1) numpy array where only last column is guaranteed non-NaN values and is the labels
    Return:
        mean_clean: N x (D+1) numpy array where each NaN value in data has been replaced by the mean feature value
    Notes: 
        (1) When taking the mean of any feature, do not count the NaN value
        (2) Return all values to max one decimal point
        (3) The labels column will never have NaN values
    """
    mean_clean = data.copy() #copy 

    for feature in range(mean_clean.shape[1] - 1):  #ft iter
        mean_value = np.nanmean(mean_clean[:, feature])  #mean
        mean_clean[:, feature][np.isnan(mean_clean[:, feature])] = mean_value   #replace NaN with mean 
    mean_clean = np.round(mean_clean, 1) 

    return mean_clean
    #raise NotImplementedError

class SemiSupervised(object):
    def __init__(self): # No need to implement
        pass
    
    def softmax(self,logit): # [0 pts] - can use same as for GMM
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array where softmax has been applied row-wise to input logit
        """
        exp_form = np.exp(logit - np.max(logit, axis=1, keepdims=True))
        exp_row_sum = np.sum(exp_form, axis=1, keepdims=True)

        prob = np.divide(exp_form , exp_row_sum)
        return prob
        
        
        #raise NotImplementedError

    def logsumexp(self,logit): # [0 pts] - can use same as for GMM
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:])
        """
        N = logit.shape[0]
        #low_dim_logit = (logit - np.max(logit, axis=1, keepdims=True)) #making it stable 
        max = np.max(logit, axis=1, keepdims=True)
        
        exp = np.exp(logit - max)
        sum_exp = np.sum(exp, axis=1)
    
        log = np.log(sum_exp).reshape((N,1))
        #sum_exp = sum_exp.reshape((N,1))
        s = log + max 
        s = s.reshape((N,1))
        return s

        #raise NotImplementedError
        #raise NotImplementedError
    
    def normalPDF(self, logit, mu_i, sigma_i): # [0 pts] - can use same as for GMM
        """
        Args: 
            logit: N x D numpy array
            mu_i: 1xD numpy array, the center for the ith gaussian.
            sigma_i: 1xDxD numpy array, the covariance matrix of the ith gaussian.  
        Return:
            pdf: 1xN numpy array, the probability distribution of N data for the ith gaussian
            
        Hint: 
            np.diagonal() should be handy.
        """
        N, D = logit.shape[0], logit.shape[1]
        
        T1  = 1/ np.sqrt(2 * np.pi * np.diag(sigma_i).reshape((1,-1)))
        T2 = np.exp(-0.5 * np.divide(np.square(logit - mu_i),np.diag(sigma_i).reshape((1,-1))))
        return np.prod(T1*T2, axis=1)
        #raise NotImplementedError
        
        #raise NotImplementedError
        
####### TO BE CHANGED ########
    
    def _init_components(self, points, K, **kwargs): # [5 pts] - modify from GMM
        """
        Args:
            points: Nx(D+1) numpy array, the observations
            K: number of components
            kwargs: any other args you want
        Return:
            pi: numpy array of length K; contains the prior probabilities of each class k
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        Hint:
            As explained in the algorithm, you need to calculate the values of mu, sigma and pi based on the labelled dataset
        """
        np.random.seed(5) #Do Not Remove Seed
        # pi = [self.K]*self.K 
        # pi = np.array(pi)
        pi = np.array([1/self.K]*self.K)
        #return pi 
        
        
        min = np.min(self.points, axis=0)
        max = np.max(self.points, axis =0 )
        s = (self.K, self.D)
        mu = np.random.uniform(min, max, s)
        #return mu 
        
        
        sigma = np.zeros((self.K,self.D,self.D))
        for x in range(self.K):
            identity = np.eye(self.D)
            sigma[x,:,:] = identity
        return pi, mu, sigma

        #raise NotImplementedError

    def _ll_joint(self, points, pi, mu, sigma, **kwargs): # [0 pts] - can use same as for GMM
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        Return:
            ll(log-likelihood): NxK array, where ll(i, j) = log pi(j) + log NormalPDF(points_i | mu[j], sigma[j])
        """
        # === graduate implementation
        #if full_matrix is True:
            #...
        len_pi = len(pi)
        #self.K=K
        x = np.log(pi + LOG_CONST)  #mitigate run time error 
        ll = np.zeros((self.N, len_pi))
    
        #if full_matrix ==  True: 
        for n in range(self.K):
                #def multinormalPDF(self, points, mu_i, sigma_i):  # [5pts]   doing 
            density_fun = self.normalPDF(self.points,mu[n],sigma[n])
            y = np.log(density_fun +LOG_CONST) + x[n]
            ll[:, n] = y 
        return ll 
    
    def _E_step(self, points, pi, mu, sigma, **kwargs): # [0 pts] - can use same as for GMM
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        Return:
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            
        Hint: You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above. 
        """
        ll = self._ll_joint(pi, mu, sigma)
        gamma = self.softmax(ll)
        return gamma
        raise NotImplementedError

    def _M_step(self, points, gamma, **kwargs): # [0 pts] - can use same as for GMM
        """
        Args:
            points: NxD numpy array, the observations
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. 
            
        Hint:  There are formulas in the slide.
        """
        #if full_matrix is True: 
        pi = np.sum(gamma, axis=0) / gamma.shape[0] #pi done 
        sigma = np.zeros((gamma.shape[1], self.points.shape[1], self.points.shape[1]))
        mu = np.zeros((gamma.shape[1], self.points.shape[1]))
        for i in range(gamma.shape[1]):
        #gamma[:, i][:, np.newaxis]
            c1 = gamma[:,i]
            c2 = c1[:,np.newaxis]
            c_sum = np.sum(c2)
            #k_mu_sum = np.sum(self.points * c1, axis =0) 
            k_mu_sum = np.sum(self.points * c1[:, np.newaxis], axis=0) / c_sum
            #k_mu_sum = np.divide(k_mu_sum, c_sum)
            mu[i,:] = k_mu_sum
                
            #T1 = (c * (self.points - k_mu_sum))
            T1 = c1[np.newaxis,:] * (self.points -k_mu_sum).T
            T2 = np.matmul(T1, (self.points - k_mu_sum)) / c_sum 
            sigma[i, :, :] = T2 
        return pi,mu,sigma
    
    
    
####### TO BE CHANGED ########
    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, **kwargs): # [5 pts] - modify from GMM
        """
        Args:
            points: N x (D+1) numpy array, where 
                - N is # points, 
                - D is the number of features,
                - the last column is the point labels (when available) or NaN for unlabeled points
            K: integer, number of clusters
            max_iters: maximum number of iterations
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want
        Return:
            pi, mu, sigma: (1xK np array, KxD numpy array, KxDxD numpy array)
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))

        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma)

            # M-step
            pi, mu, sigma = self._M_step(gamma)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return  pi, mu, sigma
       # raise NotImplementedError


class ComparePerformance(object):

    def __init__(self): #No need to implement
        pass

    @staticmethod
    def accuracy_semi_supervised(training_data, validation_data, K:int) -> float: # [2.5 pts]
        """
        Train a classification model using your SemiSupervised object on the training_data.
        Classify the validation_data using the trained model
        Return the accuracy score of the model's predicted classification of the validation_data 

        Args:
            training_data: N_t x (D+1) numpy array, where 
                - N_t is the number of data points in the training set, 
                - D is the number of features, and 
                - the last column represents the labels (when available) or a flag that allows you to separate the unlabeled data.
            validation_data: N_v x(D+1) numpy array, where 
                - N_v is the number of data points in the validation set,
                - D is the number of features, and 
                - the last column are the labels
            K: integer, number of clusters for SemiSupervised object
        Return:
            accuracy: floating number
        
        Note: (1) validation_data will NOT include any unlabeled points
              (2) you may use sklearn accuracy_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
        """


        #raise NotImplementedError

    @staticmethod
    def accuracy_GNB(training_data, validation_data) -> float: # [2.5 pts]
        """
        Train a Gaussion Naive Bayes classification model (sklearn implementation) on the training_data.
        Classify the validation_data using the trained model
        Return the accuracy score of the model's predicted classification of the validation_data 

        Args:
            training_data: N_t x (D+1) numpy array, where 
                - N is the number of data points in the training set, 
                - D is the number of features, and 
                - the last column represents the labels
            validation_data: N_v x (D+1) numpy array, where 
                - N_v is the number of data points in the validation set,
                - D is the number of features, and 
                - the last column are the labels
        Return:
            accuracy: floating number

        Note: (1) both training_data and validation_data will NOT include any unlabeled points
              (2) use sklearn implementation of Gaussion Naive Bayes: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
        """
        X_train = training_data[:, :-1]
        X_valid = validation_data[:, :-1]
        y_train = training_data[:, -1]
        y_valid = validation_data[:, -1]
        
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        y_pred = gnb.predict(X_valid)
        
        accuracy = accuracy_score(y_valid, y_pred)
        return accuracy

        #raise NotImplementedError







### Self Notes ### 
        # n_inc = incomplete_points.shape[0]  # Number of incomplete points
        # n_comp = complete_points.shape[0]   # Number of complete points
        # num_features = incomplete_points.shape[1] - 1  # Number of features (excluding the class label)
        # d = complete_points.shape[1]

        # # clean_points = np.copy(incomplete_points)

        # # for f_idx in range(num_features):
        # #         for i in range(n_inc):
        # #             match_labs = complete_points[:, -1] == incomplete_points[i, -1]
        # #             inc_feat_value = incomplete_points[i, f_idx]

        # #             match = complete_points[match_labs]
        # #             nn = []

        # #             for dp in match:
        # #                 distance = np.linalg.norm(incomplete_points[i, :-1] - dp[:-1])
        # #                 nn.append((distance, dp[f_idx]))
        # #                 nn.sort(key=lambda x: x[0])
        # #                 nn = nn[:K]
        # #                 avg_feature_value = np.mean([neighbor[1] for neighbor in nn])
        # #                 clean_points[i, f_idx] = avg_feature_value

        # # # return clean_points
        # # for i in range(n_inc): 
        # #     #k_near = np.argpartition(self.pairwise_dist([i], K))[:K]
        # #     k_near = np.argpartition(self.pairwise_dist(i, K))[:K]
        # #     for f_inx in range(d -1):
        # #         feat = n_comp[k_near, f_inx]
        # #         mean = np.nanmean(feat)
        # #         n_inc[i, f_inx] = mean 
        
        # # clean_points = np.vstack([n_inc, n_comp])
        # # return clean_points 
            
            
            
            