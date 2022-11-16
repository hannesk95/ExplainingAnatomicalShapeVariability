import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigsh as ssl_eigsh
from time import clock

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn import utils
from sklearn.preprocessing import KernelCenterer, scale
from sklearn.metrics.pairwise import pairwise_kernels

# REMARK: 
# Code is based on the paper: 
# "Supervised principal component analysis: Visualization, classification and regression on subspaces and submanifolds"
# and was adopted from:
# https://github.com/kumarnikhil936/supervised_pca.git

class SPCA(BaseEstimator, TransformerMixin):
    
    def __init__(self, num_components, kernel="linear", eigen_solver='auto', 
                 max_iterations=None, gamma=0, degree=3, coef0=1, alpha=1.0, 
                 tolerance=0, fit_inverse_transform=False):
        
        self._num_components = num_components
        self._gamma = gamma
        self._tolerance = tolerance
        self._fit_inverse_transform = fit_inverse_transform
        self._max_iterations = max_iterations
        self._degree = degree
        self._kernel = kernel
        self._eigen_solver = eigen_solver
        self._coef0 = coef0
        self._centerer = KernelCenterer()
        self._alphas = alpha
        
        
    def _get_kernel(self, X, Y=None):
        # Returns a kernel matrix K such that K_{i, j} is the kernel between the ith and jth vectors 
        # of the given matrix X, if Y is None. 
        
        # If Y is not None, then K_{i, j} is the kernel between the ith array from X and the jth array from Y.
        
        # valid kernels are 'linear, rbf, poly, sigmoid, precomputed'

        print("Hannes")
        Y = np.repeat(Y, 6144, axis=1)
        
        args = {"gamma": self._gamma, "degree": self._degree, "coef0": self._coef0}
        
        return pairwise_kernels(X, Y, metric=self._kernel, n_jobs=-1, filter_params=True, **args)
    
    
    
    def _fit(self, X, Y):

        n_samples, n_features = X.shape
        
        # calculate kernel matrix of the labels Y and centre it and call it K (=H.L.H)
        K = self._centerer.fit_transform(self._get_kernel(X, Y))
        
        # deciding on the number of components to use
        if self._num_components is not None:
            num_components = min(K.shape[0], self._num_components)
        else:
            num_components = self.K.shape[0]
        
        # Scale X
        # scaled_X = scale(X)
        
        # calculate the eigen values and eigen vectors for X^T.K.X
        Q = (X.T).dot(K).dot(X)
        
        # If n_components is much less than the number of training samples, 
        # arpack may be more efficient than the dense eigensolver.
        if (self._eigen_solver=='auto'):
            if (Q.shape[0]/num_components) > 20:
                eigen_solver = 'arpack'
            else:
                eigen_solver = 'dense'
        else:
            eigen_solver = self._eigen_solver
        
        if eigen_solver == 'dense':
            # Return the eigenvalues (in ascending order) and eigenvectors of a Hermitian or symmetric matrix.
            self._lambdas, self._alphas = linalg.eigh(Q, eigvals=(Q.shape[0] - num_components, Q.shape[0] - 1))
            # argument eigvals = Indexes of the smallest and largest (in ascending order) eigenvalues
        
        elif eigen_solver == 'arpack':
            # deprecated :: self._lambdas, self._alphas = utils.arpack.eigsh(A=Q, num_components, which="LA", tol=self._tolerance)
            self._lambdas, self._alphas = ssl_eigsh(A=Q, k=num_components, which="LA", tol=self._tolerance)
            
        indices = self._lambdas.argsort()[::-1]
        
        self._lambdas = self._lambdas[indices]
        self._lambdas = self._lambdas[self._lambdas > 0]  # selecting values only for non zero eigen values
        
        self._alphas = self._alphas[:, indices]
        self._alphas = self._alphas[:, self._lambdas > 0]  # selecting values only for non zero eigen values
        
        self.X_fit = X

        
        explained_variance_ = (self._lambdas**2) / (n_samples - 1)
        total_var = explained_variance_.sum()
        self.explained_variance_ratio_ = explained_variance_ / total_var

        
    def _transform(self):
        return self.X_fit.dot(self._alphas)


    def transform(self, X):
        return X.dot(self._alphas)


    def inverse_transform(self, z):
        return z.dot(self._alphas.T)


    def fit(self, X, Y):
        self._fit(X,Y)
        return


    def fit_and_transform(self, X, Y):
        self.fit(X, Y)
        return self._transform()


class Kernel_SPCA(BaseEstimator, TransformerMixin):
    
    def __init__(self, num_components, xkernel={'kernel': "sigmoid", 'gamma': 1, 'degree': 3, 'coef0': 1}, 
                 ykernel={'kernel': "linear", 'gamma': 0, 'degree': 3, 'coef0': 1}, eigen_solver='auto', 
                 max_iterations=None, gamma=0, degree=3, coef0=1, alpha=1.0, tolerance=0, fit_inverse_transform=False):
        
        self._num_components = num_components
        self._xkernel = xkernel
        self._ykernel = ykernel
        self._tolerance = tolerance
        self._fit_inverse_transform = fit_inverse_transform
        self._max_iterations = max_iterations
        self._eigen_solver = eigen_solver
        self._centerer = KernelCenterer()
        self._alphas = alpha
        
        
    def _get_kernel(self, X, args, Y=None):
        # Returns a kernel matrix K such that K_{i, j} is the kernel between the ith and jth vectors 
        # of the given matrix X, if Y is None. 
        
        # If Y is not None, then K_{i, j} is the kernel between the ith array from X and the jth array from Y.
        
        # valid kernels are 'linear, rbf, poly, sigmoid, precomputed'
               
        return pairwise_kernels(X, Y, metric=args['kernel'], n_jobs=-1, filter_params=True, **args)
    
    
    
    def _fit(self, X, Y):
        
        # calculate kernel matrix of the labels X and Y and centre it and call it Kx and Ky
        Ky = self._centerer.fit_transform(self._get_kernel(Y, self._ykernel))
        Kx = self._centerer.fit_transform(self._get_kernel(X, self._xkernel))
        
        # deciding on the number of components to use
        if self._num_components is not None:
            num_components = min(Ky.shape[0], self._num_components)
        else:
            num_components = self.Ky.shape[0]
  
        # calculate the eigen values and eigen vectors for X^T.K.X
        Q = (Kx).dot(Ky).dot(Kx)
        
        # If n_components is much less than the number of training samples, 
        # arpack may be more efficient than the dense eigensolver.
        if (self._eigen_solver=='auto'):
            if (Q.shape[0]/num_components) > 20:
                eigen_solver = 'arpack'
            else:
                eigen_solver = 'dense'
        else:
            eigen_solver = self._eigen_solver
        
        if eigen_solver == 'dense':
            # Return the eigenvalues (in ascending order) and eigenvectors of a Hermitian or symmetric matrix.
            self._lambdas, self._alphas = linalg.eigh(Q, Kx, eigvals=(Q.shape[0] - num_components, Q.shape[0] - 1))
            # argument eigvals = Indexes of the smallest and largest (in ascending order) eigenvalues
        
        elif eigen_solver == 'arpack':
            # deprecated :: self._lambdas, self._alphas = utils.arpack.eigsh(A=Q, num_components, which="LA", tol=self._tolerance)
            self._lambdas, self._alphas = ssl_eigsh(A=Q, M=Kx, k=num_components, which="LA", tol=self._tolerance)
            
        indices = self._lambdas.argsort()[::-1]
        
        self._lambdas = self._lambdas[indices]
        self._lambdas = self._lambdas[self._lambdas > 0]  # selecting values only for non zero eigen values
        
        self._alphas = self._alphas[:, indices]
        self._alphas = self._alphas[:, self._lambdas > 0]  # selecting values only for non zero eigen values
        
        self.X_fit = X
        self.Kx_fit = Kx

        
    def _transform(self):
        return self.Kx_fit.dot(self._alphas)


    def transform(self, X):
        K = self._get_kernel(self.X_fit, self._xkernel, X)
        return K.T.dot(self._alphas)


    def fit(self, X, Y):
        self._fit(X,Y)
        return


    def fit_and_transform(self, X, Y):
        self.fit(X, Y)
        return self._transform()