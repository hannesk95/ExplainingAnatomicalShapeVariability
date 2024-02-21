import numpy as np
from scipy.linalg import svd
from scipy.sparse.linalg import eigsh

class SPCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, Y):
        '''
        X = (features, samples)
        y = (label, samples)
        '''

        n_features, n_samples = X.shape

        H = np.eye(X.shape[1])-1/X.shape[0] * np.ones(X.shape[1]) @ np.ones(X.shape[1]).T
        phi = X @ H @ Y.T
        PHI = phi @ phi.T
        U, S, Vt = svd(PHI)

        self.U = U[:,:self.n_components]
        self.S = S[:self.n_components]
        self.H = H

        # egnvalues = self.S
        total_egnvalues = sum(self.S)
        self.explained_variance_ratio_ = [(i/total_egnvalues) for i in sorted(self.S, reverse=True)]


        # explained_variance_ = (self.U**2) / (n_samples - 1)
        # total_var = explained_variance_.sum()
        # self.explained_variance_ratio_ = explained_variance_ / total_var


    def transform(self, X):
        return self.U.T @ X

    def inverse_transform(self, Z):
        #inv = np.linalg.pinv(self.U)
        return (self.U @ Z @ self.H).T
