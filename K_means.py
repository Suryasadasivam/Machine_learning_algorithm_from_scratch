import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KMEANS():
    def __init__(self,K=5,max_iter=1000,plot_steps=False):
        self.k=K
        self.max_iters=max_iter
        self.plot_steps=plot_steps
        
        self.clusters=[[] for _ in range(self.k)]
        self.centroids=[]
    def predict(self, x):
        self.x=x
        self.n_samples,self.n_features=x.shape
        
        #we need to initialise the centriod. create a random number within n_sample range 
        random_idx=np.random.choice(self.n_samples,self.k,replace=False)
        #selecting a random points as centroid in  give data 
        self.centroids=[self.x[idx] for idx in random_idx]

        
        #optimization
        for _ in range(self.max_iters):
            # assign a samples to centroid 
            self.clusters=self.createcluster(self.centroids)
            
            if self.plot_steps:
                self.plot()
            #update new centroids from cluster
            centroids_old=self.centroids
            
            self.centroids=self.get_centroids(self.clusters) 
            
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()
        return self._get_cluster_labels(self.clusters)
            
    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels        
            
    def createcluster(self, centroids):
        clusters=[[] for _ in range(self.k)]
        for idx,sample in enumerate(self.x):
            centroid_idx=self.closest_centroid(sample,centroids)
            clusters[centroid_idx].append(idx)
        return clusters
    
    def closest_centroid(self,sample,centroids):
        distance=[euclidean_distance(sample,point) for point in centroids]
        closest_ind=np.argmin(distance)
        return closest_ind
    
    def get_centroids(self,clusters):
        centroids=np.zeros((self.k ,self.n_features))
        for clusters_idx,cluster in enumerate(clusters):
            cluster_mean = np.mean(self.x[cluster], axis=0)
            centroids[clusters_idx] = cluster_mean
        return centroids
    
    def _is_converged(self, centroids_old, centroids):
    # distances between each old and new centroids, for all centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.k)]
        return sum(distances) == 0   
    
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.x[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show() 
            