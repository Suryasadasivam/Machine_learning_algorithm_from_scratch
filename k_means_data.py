from sklearn.datasets import make_blobs
from K_means import KMEANS
import numpy as np

X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40)
print(X.shape)

clusters = len(np.unique(y))
print(clusters)

k = KMEANS(K=clusters, max_iter=150, plot_steps=True)
y_pred = k.predict(X)

k.plot()