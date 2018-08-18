# Cluster Stability

Python implementation of the following paper:

Lange, T., BraunRoth, V., Braun, M., & Buhmann, J. (2004) Stability-based validation of clustering solutions. In *Neural Comput*. 2004 Jun;16(6):1299-323. massachusetts institute of Technology


Goal is to identify the "true" number of clusters in a dataset by determining how stable clusters are when generated at each value of k based on the hamming distance metric.


My implementation seems to favor lower values of k, which should be handled by the dividing the stability values by the random stability values, but isn't right now. Not totally sure if this is a result of the dataset and classifier or an actual bug, but something to be aware of if you're using this.

## Example

```
from cluster_stability import ClusterStability
from sklearn import datasets
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import RandomForestClassifier

data = datasets.make_blobs(n_samples=n_samples, centers=4)[0]
k_values = [2, 3, 4, 5, 6]
cluster_method = AgglomerativeClustering
cluster_params = None
model = RandomForestClassifier
num_reps = 5
param_grid = None

cs = ClusterStability(data=data,
                      k_values=k_values,
                      cluster_method=cluster_method,
                      cluster_params=cluster_params,
                      model=model,
                      num_reps=num_reps,
                      param_grid=param_grid)
cs.fit()
cs.plot_values()
```
