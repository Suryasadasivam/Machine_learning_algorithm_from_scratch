import numpy as np 

class svm():
    def __init__(self,learning_rate=0.001,lamda_param=0.01,n_iters=1000):
        self.lr=learning_rate
        self.lamda_param=lamda_param
        self.n_iters=n_iters
        self.weight=None
        self.bias=None
    def fit(self,x,y):
        y_=np.where(y<=0,-1,1)
        n_sample,n_feature=x.shape
        self.weight=np.zeros(n_feature)
        self.bias=0
        
        for _ in range(self.n_iters):
            for idx,x_i in enumerate(x):
                condition=y_[idx]*(np.dot(x_i,self.weight)+self.bias)>=1
                if condition:
                    self.weight-=self.lr*(2*self.lamda_param*self.weight)
                else:
                    self.weight-=self.lr*(2*self.lamda_param*self.weight)*np.dot(x_i,y_[idx])
                    self.bias-=self.lr*y_[idx]
    def predict(self,x_test):
        linear=np.dot(x_test,self.weight)+self.bias
        return linear
            
        
         