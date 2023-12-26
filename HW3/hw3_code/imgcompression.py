import numpy as np
from typing import Tuple


class ImgCompression(object):
    def __init__(self):
        pass

    def svd(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # [4pts]
        """
        Do SVD. You could use numpy SVD.
        Your function should be able to handle black and white
        images ((N,D) arrays) as well as color images ((3,N,D) arrays)
        In the image compression, we assume that each column of the image is a feature. Perform SVD on the channels of
        each image (1 channel for black and white and 3 channels for RGB)
        Image is the matrix X.

        Hint: np.linalg.svd by default returns the transpose of V. We want you to return the transpose of V, not V.

        Args:
            X: (N,D) numpy array corresponding to black and white images / (3,N,D) numpy array for color images

        Return:
            U: (N,N) numpy array for black and white images / (3,N,N) numpy array for color images
            S: (min(N,D), ) numpy array for black and white images / (3,min(N,D)) numpy array for color images
            V^T: (D,D) numpy array for black and white images / (3,D,D) numpy array for color images
        """
        ####
        '''Notes for self: 
         The left singular matrix, U, contains the principal components of the data. The right singular matrix, V, contains the directions of 
         maximum variance in the data. The singular values in S represent the importance of each principal component.
        '''
        x_2d = X.shape[0]
        f = X.shape[1]
        if len(X.shape) == 2:
            U,S,V = np.linalg.svd(X, full_matrices = True)
            return U,S,V

        
        # x_3d = X.shape[2]
        # #print(x_3d)

        # U = np.zeros((3,f,f))
        # V = np.zeros((f,f,x_3d))
        # s = min(X.shape[1],X.shape[2])
        # S = np.zeros((s,x_3d))
        # for i in range(x_3d) : 
        #     x = X[:,:,i]
        #     u,s,v = np.linalg.svd(x, full_matrices = True)
        #     U[:,:,i] = u
        #     V[:,:,i] = v   
        #     S[:,i] = s

        else:  #col 3d
            U = np.zeros((3, X.shape[1], X.shape[1]))
            S = np.zeros((3, min(X.shape[1], X.shape[2])))
            VT = np.zeros((3, X.shape[2], X.shape[2]))
            for i in range(3):
                U0, S0, VT0 = np.linalg.svd(X[i, :, :], full_matrices=True)
                U[i, :, :] = U0
                S[i, :] = S0
                VT[i, :, :] = VT0

        return U, S, VT

    
    def compress(
        self, U: np.ndarray, S: np.ndarray, V: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # [4pts]
        """Compress the SVD factorization by keeping only the first k components

        Args:
            U (np.ndarray): (N,N) numpy array for black and white simages / (3,N,N) numpy array for color images
            S (np.ndarray): (min(N,D), ) numpy array for black and white images / (3,min(N,D)) numpy array for color images
            V (np.ndarray): (D,D) numpy array for black and white images / (3,D,D) numpy array for color images
            k (int): int corresponding to number of components to keep

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                U_compressed: (N, k) numpy array for black and white images / (3, N, k) numpy array for color images
                S_compressed: (k, ) numpy array for black and white images / (3, k) numpy array for color images
                V_compressed: (k, D) numpy array for black and white images / (3, k, D) numpy array for color images
        """
        # N= U.shape[0]
        # D = V.shape[0]
        # #2d images 
        # if len(U.shape) == 2:
        #     #keep first k 
        #     U_compressed =  U[:,:k]
     
        #     S_compressed = S[:k,:]
        #     V_compressed = V[:k,:,:]
        #     return U_compressed,S_compressed,V_compressed
        if (len(U.shape) == 3) :
            return U[:, :, :k], S[:, :k], V[:, :k, :]
        else:
            return U[:, :k], S[:k], V[:k, :]

        #raise NotImplementedError

    def rebuild_svd(
        self,
        U_compressed: np.ndarray,
        S_compressed: np.ndarray,
        V_compressed: np.ndarray,
    ) -> np.ndarray:  # [4pts]
        """
        Rebuild original matrix X from U, S, and V which have been compressed to k componments.

        Args:
            U_compressed: (N,k) numpy array for black and white images / (3,N,k) numpy array for color images
            S_compressed: (k, ) numpy array for black and white images / (3,k) numpy array for color images
            V_compressed: (k,D) numpy array for black and white images / (3,k,D) numpy array for color images

        Return:
            Xrebuild: (N,D) numpy array of reconstructed image / (3,N,D) numpy array for color images

        Hint: numpy.matmul may be helpful for reconstructing color images
        """
        D = V_compressed.shape[1]
        N = U_compressed.shape[0]
        k = S_compressed.shape[0]
        #black and white 

        if len(U_compressed.shape) == 2: 
            diagonal_S = np.diag(S_compressed)     #diagonal of S matrix 
            s_dg_Vcomp_2d = np.dot(diagonal_S, V_compressed)      #multuply diagnoal_S and V_compressed
            Xrebuild_2d =  np.dot(U_compressed, s_dg_Vcomp_2d) #rebuild 
            return Xrebuild_2d

        N_3d = V_compressed.shape[2] #channels for colour image 
        rebuild_3d = np.zeros((3, U_compressed.shape[1], N_3d))

        for x in range(3): 
            u = U_compressed[x,:,:]
            s = S_compressed[x,:]
            v = V_compressed[x,:,:]
            us = (u*s)
            rebuild_3d[x,:,:] = np.dot(us, v)

            # v = V_compressed[x, :, :]
            # s = S_compressed[x, :]
            # u = U_compressed[x, : , :]

            # diagonal_S= np.diag(s)
            # s_dg_Vcomp_3d = np.dot(diagonal_S, v)
            # i = np.dot(u,s)
            # rebuild_3d[x, : , :] = np.dot(i, v)

            #reconstruction 
            #rebuild_3d = np.dot(u, np.dot(diagonal_S, v))
        return rebuild_3d


    def compression_ratio(self, X: np.ndarray, k: int) -> float:  # [4pts]
        """
        Compute the compression ratio of an image: (num stored values in compressed)/(num stored values in original)

        Args:
            X: (N,D) numpy array corresponding to black and white images / (3,N,D) numpy array for color images
            k: int corresponding to number of components

        Return:
            compression_ratio: float of proportion of storage used by compressed image
        """
        if len(X.shape) == 2: #bw img
            org2d = X.shape[0] * X.shape[1]
            compressed2d = (k + (X.shape[0] *k) + (X.shape[1] *k))
            ratio2d = compressed2d/org2d
            return ratio2d
        compressed3d =  ((X.shape[1]*k)+(X.shape[2]*k)+k)  #(N*k+D*k+k)
        original3d = (X.shape[1] * X.shape[2])
        return compressed3d/original3d



        #raise NotImplementedError

    def recovered_variance_proportion(self, S: np.ndarray, k: int) -> float:  # [4pts]
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
           S: (min(N,D), ) numpy array black and white images / (3,min(N,D)) numpy array for color images
           k: int, rank of approximation

        Return:
           recovered_var: float (array of 3 floats for color image) corresponding to proportion of recovered variance
        """

        ssq = (S**2)

        if len(S.shape) ==1: 
            x = np.sum(ssq[:k])
            y = np.sum(ssq)
            return x/y
        
        x = S.shape[1]
        var = np.zeros(3)

        for i in range(3):
            var[i] = np.sum(S[i, :k]**2) / np.sum(S[i, :]**2)
        return var
        #raise NotImplementedError

    def memory_savings(
        self, X: np.ndarray, U: np.ndarray, S: np.ndarray, V: np.ndarray, k: int
    ) -> Tuple[int, int, int]:
        """
        PROVIDED TO STUDENTS

        Returns the memory required to store the original image X and
        the memory required to store the compressed SVD factorization of X

        Args:
            X (np.ndarray): (N,D) numpy array corresponding to black and white images / (3,N,D) numpy array for color images
            U (np.ndarray): (N,N) numpy array for black and white simages / (3,N,N) numpy array for color images
            S (np.ndarray): (min(N,D), ) numpy array for black and white images / (3,min(N,D)) numpy array for color images
            V (np.ndarray): (D,D) numpy array for black and white images / (3,D,D) numpy array for color images
            k (int): integer number of components

        Returns:
            Tuple[int, int, int]:
                original_nbytes: number of bytes that numpy uses to represent X
                compressed_nbytes: number of bytes that numpy uses to represent U_compressed, S_compressed, and V_compressed
                savings: difference in number of bytes required to represent X
        """

        original_nbytes = X.nbytes
        U_compressed, S_compressed, V_compressed = self.compress(U, S, V, k)
        compressed_nbytes = (
            U_compressed.nbytes + S_compressed.nbytes + V_compressed.nbytes
        )
        savings = original_nbytes - compressed_nbytes

        return original_nbytes, compressed_nbytes, savings

    def nbytes_to_string(self, nbytes: int, ndigits: int = 3) -> str:
        """
        PROVIDED TO STUDENTS

        Helper function to convert number of bytes to a readable string

        Args:
            nbytes (int): number of bytes
            ndigits (int): number of digits to round to

        Returns:
            str: string representing the number of bytes
        """
        if nbytes == 0:
            return "0B"
        units = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
        scale = 1024
        units_idx = 0
        n = nbytes
        while n > scale:
            n = n / scale
            units_idx += 1
        return f"{round(n, ndigits)} {units[units_idx]}"
