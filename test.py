from cluster_stability import ClusterStability
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from numpy import dot

# Generate data
n_samples = 2000

noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)[0]
noisy_circles_k_values = [2, 3, 4, 5]

blobs_k_values = [2, 3, 4, 5, 6]
blobs_3 = datasets.make_blobs(n_samples=n_samples, centers=3)[0]
blobs_4 = datasets.make_blobs(n_samples=n_samples, centers=4)[0]
blobs_5 = datasets.make_blobs(n_samples=n_samples, centers=5)[0]

X, y = datasets.make_blobs(n_samples=n_samples, centers=3)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
aniso = dot(X, transformation)
aniso_k_values = [2, 3, 4, 5, 6]

# Initialize cluster stability class
cluster_method = SpectralClustering
cluster_params = None
model = RandomForestClassifier
num_reps = 10
param_grid = None

cs = ClusterStability(data=blobs_4,
                      k_values=blobs_k_values,
                      cluster_method=cluster_method,
                      cluster_params=cluster_params,
                      model=model,
                      num_reps=num_reps,
                      param_grid=param_grid)

cs.fit()
cs.plot_values()
