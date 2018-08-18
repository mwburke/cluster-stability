from random import randint
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import hamming
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np


class ClusterStability:
    """
    Cluster stability metric to determine "true" number of clusters
    by using repeated cluster assignment similarity metrics for a set
    of number of clusters k1..kn as described in the following paper:
    Stability-Based Validation of Clustering Solutions

    Brief algorithm description:
        D = dataset
        n = number of times to get stability metric for each k
        ks = array of number of clusters k to check stability over

        For each k in ks:
            For 1..n:
                Randomly split dataset into equal sized datasets D1, D2
                Perform clustering using input algorithm A to produce
                    cluster assignment X1 from D1, X2 from D2
                Train the classifier on X1 (using the parameter grid if not null) and use it to provide
                    predictions for D2 as X2'. Appropriate classifier choice is critical to get good
                    results and should be chosen using prior testing
                Use the Hungarian algorithm (https://en.wikipedia.org/wiki/Hungarian_algorithm)
                    as implemented in scipy.optimize.linear_sum_assignment
                    to align equivalent cluster assignments in both X1 and X2 for accurate comparison
                Calculate the normalized Hamming distance between aligned X2 and X2' as
                    the initial stability result SA
                Create two sets of data R1 and R2 the sizes of X1 and X2 respectively
                    and create cluster assignments for their values randomly with each
                    data point having an equal chance (1/k) to be assigned to cluster k
                Calculate the random stability result SR from R1 and R2 as with X1 and X2
                Calculate the stability index for this run by dividing the algorithm stability
                    by the random stability (SA / SR)
            Return the average stability value

        The k value with the lowest stability index should likely be the "true" number of clusters
    """

    def __init__(self, data, k_values, cluster_method, cluster_params, model, num_reps=3, param_grid=None):
        """
        Parameters
        ----------
            data : pandas dataframe or numpy array
                Data on which to perform cluster stability analysis
            k_values : int array, shape = [num_cluster_values]
                Number of clusters to test
            cluster_method : algorithm that implements a fit_predict method that returns
                an object data cluster assignments similar to scikit-learn clustering algorithms
            cluster_params : dict or None
                Parameters to pass to clustering algorithm to generate clusters or None if
                the default parameters should be used
            model : algorithm that implements fit and predict methods similar to scikit-learn
            num_reps : int
                Number of times to calculate stability metric per value of k
            param_grid : dict or None
                Parameters on which to train the model. None is base model should
                be used without parameter tuning
        """
        self.data = data
        self.k_values = k_values
        self.cluster_method = cluster_method
        self.cluster_params = cluster_params
        self.model = model
        self.num_reps = num_reps
        self.param_grid = param_grid

    def fit(self):
        """
        Calculate stability metrics for all values of k
        """
        self.k_stability_scores = []
        for  k in self.k_values:
            stability_scores = []
            for _ in range(self.num_reps):
                stability_scores.append(self.calculate_k_stability(k))
            self.k_stability_scores.append(stability_scores)

    def get_values(self):
        return self.k_stability_scores

    def plot_values(self):
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.boxplot(np.transpose(np.array(self.k_stability_scores)))
        ax.set_xticklabels(self.k_values)
        plt.show()

    def calculate_k_stability(self, k):
        """
        Calculate the stability metric for k clusters

        Parameters
        ----------
            k : int
                Number of clusters

        Returns
        -------
        stability_metric : float
            Stability calculation for k clusters using class clustering method
            and classifier
        """
        if self.cluster_params is None:
            cluster_model = self.cluster_method
            cluster_model = cluster_model(n_clusters=k)
        else:
            cluster_params = self.cluster_params
            cluster_params['n_clusters'] = k
            cluster_model = self.cluster_method
            cluster_model = cluster_model(**cluster_params)

        train, test = train_test_split(self.data, test_size=0.5)

        train_labels = cluster_model.fit_predict(train)
        test_labels = cluster_model.fit_predict(test)

        model = self.model()

        if self.param_grid is None:
            model.fit(train, train_labels)
        else:
            gs = GridSearchCV(estimator=model,
                                 param_grid=self.param_grid)
            gs.fit(train, train_labels)
            model = gs.best_estimator_

        pred_test_labels = model.predict(test)

        aligned_labels, aligned_pred_labels = self.align_clusters(test_labels, pred_test_labels)
        distance = hamming(aligned_labels, aligned_pred_labels)

        rand_labels_v1 = self.generate_random_clusters(k, len(train))
        rand_labels_v2 = self.generate_random_clusters(k, len(train))

        rand_distance = hamming(rand_labels_v1, rand_labels_v2)

        return distance / rand_distance

    def align_clusters(self, labels_v1, labels_v2):
        """
        Use the Hungarian (Kuhn–Munkres) algorithm to align two sets of cluster
        labels for the same dataset as phrased as a combinational optimziation
        algorithm that runs in O(n^3) time. Recreates the cluster labels for
        both versions of the cluster labels and returns them.

        Parameters
        ----------
        labels_v1 : int array : shape=[num_cluster_items]
            Labels for the first version of the cluster data
        labels_v2 : int array : shape=[num_cluster_items]
            Labels for the second version of the cluster data

        Returns
        -------
        assignments[0] : int array : shape=[num_cluster_items]
            New labels for the first version of the cluster data
        assignments[1] : int array : shape=[num_cluster_items]
            New labels for the second version of the cluster data
            that are aligned with the new labels for the first version
        """
        contingency, labels_v1, labels_v2 = self.contingency_matrix(labels_v1, labels_v2)
        contingency = contingency * -1
        assignments = linear_sum_assignment(contingency)

        new_labels_v1 = self.map_labels(assignments[0], labels_v1)
        new_labels_v2 = self.map_labels(assignments[1], labels_v2)

        return new_labels_v1, new_labels_v2

    @staticmethod
    def map_labels(label_map, labels):
        return [label_map[label] for label in labels]

    @staticmethod
    def contingency_matrix(clus1_labels, clus2_labels, eps=None):
        """
        Taken from this public implementation of the Munkres/Kuhns algorithm
        under the Apache 2 license.
        Can be found here: https://github.com/bmc/munkres

        Build a contengency matrix describing the relationship between labels.
        Parameters
        ----------
        clus1_labels : int array, shape = [n_samples]
            Ground truth class labels to be used as a reference
        clus2_labels : array, shape = [n_samples]
            Cluster labels to evaluate
        eps: None or float
            If a float, that value is added to all values in the contingency
            matrix. This helps to stop NaN propogation.
            If ``None``, nothing is adjusted.
        Returns
        -------
        contingency: array, shape=[n_classes_true, n_classes_pred]
            Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
            true class :math:`i` and in predicted class :math:`j`. If
            ``eps is None``, the dtype of this array will be integer. If ``eps`` is
            given, the dtype will be float.
        class_idx: array, shape=[n_samples]
            Array of class labels with new mappings from from 0..n_classes_true
        clusters_idx: array, shape=[n_samples]
            Array of class labels with new mappings from from 0..n_classes_pred
        """
        classes, class_idx = np.unique(clus1_labels, return_inverse=True)
        clusters, cluster_idx = np.unique(clus2_labels, return_inverse=True)
        n_classes = classes.shape[0]
        n_clusters = clusters.shape[0]
        # Using coo_matrix to accelerate simple histogram calculation,
        # i.e. bins are consecutive integers
        # Currently, coo_matrix is faster than histogram2d for simple cases
        contingency = coo_matrix((np.ones(class_idx.shape[0]),
                                (class_idx, cluster_idx)),
                                shape=(n_classes, n_clusters),
                                dtype=np.int).toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
        return contingency, class_idx, cluster_idx

    @staticmethod
    def generate_random_clusters(n_clusters, length):
        """
        Generate two random sets of cluster labels with n_clusters
        with an equal 1/n_clusters chance for each point to belong
        to a given cluster.

        Parameters
        ----------
        n_clusters : int
            Number of clusters to generate labels for
        length : int
            Number of points in to generate in each output array

        Returns
        -------
        rand_labels_v1 : int array : shape[length]
            Array of random labels for n_clusters clusters
        rand_labels_v2 : int array : shape[length]
            Array of random labels for n_clusters clusters
        """
        rand_labels = [randint(1, n_clusters) for l in range(length)]

        return rand_labels
