import numpy as np 

class LinearRegression():
    def __init__(self,lr=0.001,n_iter=1000):
        self.n_iter=n_iter
        self.learningrate=lr
        self.weight=None
        self.bias=None
    def fit(self,x,y):
        n_samples,n_feature=x.shape
        self.weight=np.zeros(n_feature)
        self.bias=0
        
        
        for _ in range(self.n_iter):
            y_predicted=np.dot(x,self.weight)+self.bias
            dw=(1/n_samples) * np.dot(x.T,(y_predicted-y))
            db=(1/n_samples)*sum(y_predicted-y)
            
            self.weight=self.weight-self.learningrate*dw
            self.bias=self.bias-self.learningrate*db
            
        
    def predict(self,x_test):
        y_pred=np.dot(x_test,self.weight)+self.bias
        return y_pred