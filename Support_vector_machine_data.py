from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Support_vector_machine import svm
X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
y = np.where(y == 0, -1, 1)

clf = svm()
clf.fit(X, y)
predictions = clf.predict(X)
print(predictions)
