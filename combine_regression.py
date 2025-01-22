import numpy as np 

class BaseModel():
    def __init__(self,Lr=0.001,n_iter=1000):
        self.LearningRate=Lr
        self.n_iter=1000
        self.weight=None
        self.bias=None
    def fit(self,x,y):
        n_sample,n_feature=x.shape
        self.weight=np.zeros(n_feature)
        self.bias=0
        
        for _ in ranage(self.n_iter):
            y_predicted=self.approximate(x,self.weight,self.bias)
            dw=(1/n_samples)*np.dot(x.T,y_predicted-y)
            db=(1/n_samples)*sum(y_predicted-y)
            
            self.weight-=self.LearningRate*dw
            self.bias-=self.LearningRate*db
    def predict(x_test):
        return self.prediction(x_test,self.weight,self.bias)
    def prediction(self,x,w,b):
        pass
    def approximate(self,x,w,b):
        pass 
    
    class LinearRegression(BaseModel):
        def approximate(self,x,w,b):
            return np.dot(x,w)+b
        def prediction(self,x,w,b):
            return np.dot(x,w)+b
    class LogisticRegression(BaseModel):
        def sigmoid(self,data):
            return 1/(1+np.exp(-data))
        def approximate(self,x,w,b):
            model=np.dot(x,w)+b
            y_pred=self.sigmoid(model)
            return y_pred
        def prediction(self,x,w,b):
            model=np.dot(x,w)+b
            y_pred=self.sigmoid(model)
            y_pred_class=[1 if i>0.5 else 0 for i in y_pred]
            return np.array(y_pred_class)
            
        