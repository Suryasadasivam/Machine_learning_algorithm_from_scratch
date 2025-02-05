import numpy as np 
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN():
    def __init__(self, k=3):
        self.k=k
    def  fit(self,x,y):
        self.x_train =x
        self.y_train=y
    def predict(self, x_test):
        y_pred=[self.predicted(x) for x in x_test]
        return np.array(y_pred)
    def predicted(self,x):
        # find the distance 
        distance =[euclidean_distance(x,x_train)for x_train in self.x_train]
        # find the k nearest distance 
        k_idx=np.argsort(distance)[:self.k]
        k_neighbour_label=[self.y_train[i] for i in k_idx]
        # majortiy vote 
        most_common=Counter(k_neighbour_label).most_common(1)
        return most_common[0][0]
