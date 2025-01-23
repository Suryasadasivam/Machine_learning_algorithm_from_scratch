import numpy as np 

class PCA():
    def __init__(self,n_components):
        self.n_components=n_components
        self.components=None
        self.mean=None
    def fit(self,x):
        self.mean=np.mean(x,axis=0)
        var=x-self.mean
        
        cov=np.cov(var.T)
        
        eigenvalues, eigenvectors=np.linalg.eig(cov)
        eigenvectors = eigenvectors.T
        idx=np.argsort(eigenvalues)[::-1]
        eigenvalues=eigenvalues[idx]
        eigenvectors=eigenvectors[idx]
        
        self.components=eigenvectors[0:self.n_components]
    def transform(self,x):
        x=x-self.mean
        return np.dot(x,self.components.T)