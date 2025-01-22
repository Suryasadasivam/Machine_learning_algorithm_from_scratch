import numpy as np

class LogisticRegression():
    def __init__(self,lr=0.001,n_iter=1000):
        self.LearningRate=lr
        self.n_iter=1000
        self.weight=None
        self.bias=None
    def sigmoid(self,data):
        return 1/(1+np.exp(-data))
    
    def fit(self,x,y):
        n_samples,n_features=x.shape
        self.weight=np.zeros(n_features)
        self.bias=0
        
        for _ in range(self.n_iter):
            model= np.dot(x,self.weight)+self.bias
            y_predicted=self.sigmoid(model)
            
            dw=(1/n_samples)*np.dot(x.T,y_predicted-y)
            db=(1/n_samples)*sum(y_predicted-y)
            
            self.weight-=self.LearningRate*dw
            self.bias-=self.LearningRate*db
    def predict(self, x_test):
        y_model=np.dot(self.weight,x_test.T)+self.bias
        y_pred=self.sigmoid(y_model)
        y_pred_class=[1 if i>0.5 else 0 for i in y_pred]
        return np.array(y_pred_class)
            
        
        