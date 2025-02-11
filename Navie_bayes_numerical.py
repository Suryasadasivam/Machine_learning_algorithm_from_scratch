import numpy as np 

class NaiveBayes():
    def __init__(self):
        self.classes=None
        self.mean=None
        self.var=None
        self.prior=None
    def fit(self,x,y):
        x_sample,x_features=x.shape
        self.classes=np.unique(y)
        n_class=len(self.classes)
        
        self.mean=np.zeros((n_class,n_features),dtype=np.float64)
        self.var=np.zeros((n_class,n_features),dtype=np.float64)
        self.prior=np.zeros(n_class,dtype=np.float64)
        
        for idx,c in enumerate(self.classes):
            x_c=x[y==c]
            self.mean[idx,:]=x_c.mean(axis=0)
            self.var[idx,:]=x_c.var(axis=0)
            self.prior[idx]=x_c.shape[0]/float(n_samples)
            
    def predict(self,x):
        y_pred=[self.predicte(x)for x in x]
        return np.array(y_pred)
    def predicte(self,x):
        posteriors=[]
        
        for idx,c in enumerate(self.classes):
            prior=np.log(self.prior[idx])
            posterior=np.sum(np.log(self.pdf(idx,x)))
            posterior=prior+posterior
            posteriors.append(posterior)
            
        return self.classes[np.argmax(posteriors)]
    def pdf(self,class_idx,x):
        mean=self.mean[class_idx]
        var=self.mean[class_idx]
        nume=np.exp(-((x-mean)**2)/(2*var))
        deno=np.sqrt(2*np.pi*var)
        return numerator/denominator 