import numpy as np
from tqdm import tqdm
from kmeans import KMeans


sigma_CONST = 1e-6
LOG_CONST = 1e-32

FULL_MATRIX = True # Set False if the covariance matrix is a diagonal matrix

class GMM(object):
    def __init__(self, X, K, max_iters=100):  # No need to change
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters

        self.N = self.points.shape[0]  # number of observations
        self.D = self.points.shape[1]  # number of features
        self.K = K  # number of components/clusters

    # Helper function for you to implement   #DOING  DONE? 
    def softmax(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        Hint:
            Add keepdims=True in your np.sum() function to avoid broadcast error. 
        """
        # low_dim_logit_exp = np.exp(logit - np.max(logit, axis=1, keepdims=True)) #making it stable 
        # #exp_form = np.exp(low_dim_logit) #exp of all i in logit
        # sum_exp = np.sum(low_dim_logit_exp, axis=1, keepdims=True) #sum of all the exp 
        # prob = np.divide(low_dim_logit_exp, sum_exp)
        # return prob
        exp_form = np.exp(logit - np.max(logit, axis=1, keepdims=True))
        exp_row_sum = np.sum(exp_form, axis=1, keepdims=True)

        prob = np.divide(exp_form , exp_row_sum)
        return prob

        

    def logsumexp(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        Hint:
            The keepdims parameter could be handy
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

    # for undergraduate student
    ########## NEED TO DO THIS FOR 3.4 ###### 
    def normalPDF(self, points, mu_i, sigma_i):  # [5pts]
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """
        N, D = points.shape[0], points.shape[1]
        
        T1  = 1/ np.sqrt(2 * np.pi * np.diag(sigma_i).reshape((1,-1)))
        T2 = np.exp(-0.5 * np.divide(np.square(points - mu_i),np.diag(sigma_i).reshape((1,-1))))
        return np.prod(T1*T2, axis=1)
        #raise NotImplementedError

    # for grad students
    def multinormalPDF(self, points, mu_i, sigma_i):  # [5pts]   doing 
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            normal_pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            1. np.linalg.det() and np.linalg.inv() should be handy.
            2. The value in self.D may be outdated and not correspond to the current dataset,
            try using another method involving the current arguments to get the value of D
        """
        inverse = sigma_i.copy() #cope for ref in the try except 
        
        
        N, D = points.shape[0], points.shape[1]
        # # PDF(x) = (1 / sqrt((2 * pi)^d * det(sigma))) * exp(-0.5 * (x - mu)^T sigma^-1 (x - mu))

        try: 
            inverse = np.linalg.inv(sigma_i)
        except np.linalg.LinAlgError: 
            inverse = np.linalg.inv(inverse + sigma_CONST)
        '''
        Note to self: this solves it apparenlty. Do it if nothing works 
           https://realpython.com/python-exceptions/#the-try-and-except-block-handling-exceptions
        '''


        # ##num##
        # #x = np.sum(np.linalg.inv(sigma_i)*(points - mu_i)*(points - mu_i), axis=1)
        # # x = np.sum(np.linalg.inv(sigma_i) @ ((points - mu_i)[:, :, np.newaxis]) * (points - mu_i)[:, np.newaxis, :], axis=(1, 2))
        # # x = (-0.5)* x
        
        T1 = np.dot((points - mu_i), inverse)
        T1 = T1.T * (points - mu_i).T #dim issues (does taking a transpose fix it?)
        sum_T1 = np.sum(T1, axis =0)
        exp = np.exp(-0.5* sum_T1 )
    
        # # ##den## 
        # # y = (np.linalg.det(sigma_i) * (2 * np.pi)**d) 
        sqrt = np.sqrt(np.linalg.det(inverse))
        
        
        # # ## normal multinormal thing 
        (1/ ((2*np.pi) ** (D/2))) * sqrt * exp


        return (1/ ((2*np.pi) ** (D/2))) * sqrt * exp        
        #raise NotImplementedError


    def create_pi(self):
        """
        Initialize the prior probabilities 
        Args:
        Return:
        pi: numpy array of length K, prior
        """
        
        pi = [1/self.K]*self.K 
        pi = np.array(pi)
        return pi 

        #return NotImplementedError

    def create_mu(self):
        """
        Intialize random centers for each gaussian
        Args:
        Return:
        mu: KxD numpy array, the center for each gaussian.
        """
        
        #mu = np.random.uniform(low=np.min(self.points, axis=0), high=np.max(self.points, axis=0), size=(self.K, self.D))
        min = np.min(self.points, axis=0)
        max = np.max(self.ppints, axis =0 )
        s = (self.K, self.D)
        mu = np.random.uniform(min, max, s)
        return mu 

        #return NotImplementedError
    def create_sigma(self):
        """
        Initialize the covariance matrix with np.eye() for each k. For grads, you can also initialize the 
        by K diagonal matrices.
        Args:
        Return:
        sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
            You will have KxDxD numpy array for full covariance matrix case
        """
        sigma = np.zeros((self.K,self.D,self.D))
        for i in range(self.K):
            identity = np.eye(self.D)
            sigma[i,:,:] = identity
            
        return sigma

        #return NotImplementedError
    
    def _init_components(self, **kwargs):  # [5pts]

        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case

            Hint: np.random.seed(5) may be used at the start of this function to ensure consistent outputs.
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

    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):  # [10 pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.

        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """

        # === graduate implementation
        #if full_matrix is True:
            #...
        len_pi = len(pi)
        #self.K=K
        x = np.log(pi + LOG_CONST)  #mitigate run time error 
        ll = np.zeros((self.N, len_pi))
    
        if full_matrix ==  True: 
            for n in range(self.K):
                #def multinormalPDF(self, points, mu_i, sigma_i):  # [5pts]   doing 
                density_fun = self.multinormalPDF(self.points,mu[n],sigma[n])
                y = np.log(density_fun +LOG_CONST) + x[n]
                ll[:, n] = y 
        return ll 
        

        # === undergraduate implementation
        #if full_matrix is False:
            # ...
        #raise NotImplementedError

    def _E_step(self, pi, mu, sigma, full_matrix = FULL_MATRIX , **kwargs):  # [5pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assigmanment) for each observation.

        Hint:
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        # === graduate implementation
        #if full_matrix is True:
            # ...

        # === undergraduate implementation
        #if full_matrix is False:
            # ...
        #def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):
        ll = self._ll_joint(pi, mu, sigma, full_matrix)
        gamma = self.softmax(ll)
        return gamma
    
            
        #def softmax(self, logit): 
        
        #return self.softmax(ll)
        
        #return tau 

        #raise NotImplementedError

    def _M_step(self, gamma, full_matrix=FULL_MATRIX, **kwargs):  # [10pts]
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assigmanment) for each observation.
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case

        Hint:
            There are formulas in the slides and in the Jupyter Notebook.
            Undergrads: To simplify your calculation in sigma, make sure to only take the diagonal terms in your covariance matrix
        """
        # === graduate implementation
        #if full_matrix is True:
            # ...

        # === undergraduate implementation
        #if full_matrix is False:
            # ...
        # if full_matrix is True: 
        #     #N, K, D = gamma.shape[0], gamma.shape[1], self.points.shape[1]
            #points = self.points
        if full_matrix is True: 
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







        

        #raise NotImplementedError

    def __call__(self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=1e-16, **kwargs):  # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assigmanment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))

        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma, full_matrix)

            # M-step
            pi, mu, sigma = self._M_step(gamma, full_matrix)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)







'''
My understanding, copy to someplace safe post HW: 
GMM 
- Assumed the data fed into the model is the output of different Gaussian Dists 
- Multimodal data, several distinct data groups 
- Uses EM and soft max function 
Soft max fn 
- takes the output of the ML model and converts it into a format that easier for humans to understand.
- Vector as the input, prob dist as the output 
- for every i in the unput vector --> i ^ e / sum of all i^e  --> prob. Find the highest prob 
- used for multi class classification 
'''