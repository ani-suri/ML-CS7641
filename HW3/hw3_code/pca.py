import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from matplotlib import pyplot as plt

# Set plotly renderer
rndr_type = "jupyterlab+png"
pio.renderers.default = rndr_type


class PCA(object):
    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X: np.ndarray) -> None:  # 5 points
        """
        Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X
        You may use the numpy.linalg.svd function
        Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
        corresponding values from PCA. See the docstrings below for the expected shapes of U, S, and V transpose

        Hint: np.linalg.svd by default returns the transpose of V
              Make sure you remember to first center your data by subtracting the mean of each feature.

        Args:
            X: (N,D) numpy array corresponding to a dataset

        Return:
            None

        Set:
            self.U: (N, min(N,D)) numpy array
            self.S: (min(N,D), ) numpy array
            self.V: (min(N,D), D) numpy array
        """

        c = X - np.mean(X,axis =0 )

        u,s,v = np.linalg.svd(c, full_matrices = False)
        self.V = v
        self.S = s
        self.U = u
        #aise NotImplementedError

    def transform(self, data: np.ndarray, K: int = 2) -> np.ndarray:  # 2 pts
        """
        Transform data to reduce the number of features such that final data (X_new) has K features (columns)
        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: (N,D) numpy array corresponding to a dataset
            K: int value for number of columns to be kept

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data

        Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
        """
        #raise NotImplementedError
        v = self.V.T
        first_k = v[:,:K]
        c = data - np.mean(data,axis =0 )
        X_new = np.dot(c,first_k)
        # print(res)
        return X_new

    def transform_rv(
        self, data: np.ndarray, retained_variance: float = 0.99
    ) -> np.ndarray:  # 3 pts
        """
        Transform data to reduce the number of features such that the retained variance given by retained_variance is kept
        in X_new with K features
        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: (N,D) numpy array corresponding to a dataset
            retained_variance: float value for amount of variance to be retained

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
                   to be kept to ensure retained variance value is retained_variance

        Hint: Make sure you remember to first center your data by subtracting the mean of each feature.

        """
        #raise NotImplementedError
        ssq = self.S**2
        e_vec = self.V.T
        #self note: do not use v for variance 
        x_c = data - np.mean(data,axis =0 )

        var = ssq/np.sum(ssq)
        v_l = var.shape[0]
        
        total_v = 0
        i = 0
        while i < v_l and total_v < retained_variance:
            total_v+=var[i]
            i+=1
        
        return np.dot(x_c,e_vec)[:, :i]


    def get_V(self) -> np.ndarray:
        """Getter function for value of V"""


        return self.V

    def visualize(self, X: np.ndarray, y: np.ndarray, fig_title) -> None:  # 5 pts
        """
        You have to plot two different scatterplots (2d and 3d) for this function. For plotting the 2d scatterplot, use your PCA implementation to reduce the dataset to only 2 features. You'll need to run PCA on the dataset and then transform it so that the new dataset only has 2 features.
        Create a scatter plot of the reduced data set and differentiate points that have different true labels using color using plotly.
        Hint: Refer to https://plotly.com/python/line-and-scatter/ for making scatter plots with plotly.
        Hint: We recommend converting the data into a pandas dataframe before plotting it. Refer to https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html for more details.

        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,) numpy array, the true labels

        Return: None
        """

        #raise NotImplementedError
        self.fit(X)
        c_scheme = {"0": 'red', "1": 'blue', "2": 'black'}
        pca_x = self.transform(X,2)

        y0 = np.where(y == 0)
        cluster1 = plt.scatter(pca_x[y0,0],pca_x[y0,1],color = c_scheme["0"],marker='x',label='0') #red

        y1 = np.where(y == 1)
        cluster2 = plt.scatter(pca_x[y1,0],pca_x[y1,1],color = c_scheme["1"],marker='o',label='1') #blue

        y2 = np.where(y == 2)
        cluster3 = plt.scatter(pca_x[y2,0],pca_x[y2,1],color = c_scheme["2"],marker='*',label='2') #black 
        plt.legend()
        #legend to show and plot 
        plt.show()






