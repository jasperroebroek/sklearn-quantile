# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=True

# Authors: Jasper Roebroek <roebroek.jasper@gmail.com>
# License: BSD 3 clause

"""
Module providing the special case for quantile regression: predicting the maximum.
It only has a single implementation called RandomForestMaxRegression, which has the
same parameters as the regular RandomForestRegressor
"""
import threading

import joblib
from cython.parallel cimport prange
cimport openmp
cimport numpy as np
from joblib import Parallel

import numpy as np
from sklearn.utils.fixes import delayed, _joblib_parallel_args
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.ensemble._base import _partition_estimators
from sklearn.ensemble._forest import _generate_sample_indices
from sklearn.ensemble import RandomForestRegressor


ctypedef np.npy_intp SIZE_t              # Type for indices and counters


__all__ = ["RandomForestMaximumRegressor"]


cdef void _maximum_per_leaf(SIZE_t[::1] leaves,
                            SIZE_t[::1] unique_leaves,
                            float[::1] values,
                            SIZE_t[::1] idx,
                            float[::1] sampled_values,
                            SIZE_t n_jobs):
    """
    Index of maximum value for each unique leaf

    Parameters
    ----------
    leaves : array, shape = (n_samples)
        Leaves of a Regression tree, corresponding to weights and indices (idx)
    unique_leaves : array, shape = (n_unique_leaves)
    values : array, shape = (n_samples)
    idx : array, shape = (n_samples)
        Indices of original observations. The output will drawn from this.
    sampled_idx : shape = (n_unique_leaves)
    n_jobs : number of threads, similar to joblib
    """
    cdef:
        int i, j
        double c_leaf
        float c_max

        int n_unique_leaves = unique_leaves.shape[0]
        int n_samples = leaves.shape[0]

        int num_threads = joblib.effective_n_jobs(n_jobs)

    for i in prange(n_samples, num_threads=num_threads, nogil=True):
        c_leaf = leaves[i]
        for j in range(n_unique_leaves):
            if unique_leaves[j] == c_leaf:
                break

        if values[i] > sampled_values[j]:
            sampled_values[j] = values[i]


def _accumulate_prediction(predict, X, out, lock):
    """
    This is a utility function for joblib's Parallel. Based on the version in sklearn.ensemble._forest
    """
    prediction = predict(X, check_input=False)
    with lock:
        out[:] = np.maximum(prediction, out)


class RandomForestMaximumRegressor(RandomForestRegressor):
    """
    fit and predict functions for forest maximum regressors (quantile 1).
    Implementation based on original paper of Meinshausen (2006).
    """
    def fit(self, X, y, sample_weight=None):
        # apply method requires X to be of dtype np.float32
        X, y = check_X_y(
            X, y, accept_sparse="csc", dtype=np.float32, multi_output=False)
        super(RandomForestMaximumRegressor, self).fit(X, y, sample_weight=sample_weight)

        self.n_samples_ = len(y)

        self.y_train_ = y.astype(np.float32)
        self.y_train_leaves_ = self.apply(X).T
        self.y_weights_ = np.zeros_like(self.y_train_leaves_, dtype=np.float32)

        if sample_weight is None:
            sample_weight = np.ones(self.n_samples_)

        for i, est in enumerate(self.estimators_):
            est.y_train_ = self.y_train_
            est.y_train_leaves_ = self.y_train_leaves_[i]
            est.y_weights_ = self.y_weights_[i]
            est.verbose = self.verbose
            est.n_samples_ = self.n_samples_
            est.bootstrap = self.bootstrap
            est._i = i

        for i, est in enumerate(self.estimators_):
            if self.bootstrap:
                bootstrap_indices = _generate_sample_indices(
                    est.random_state, self.n_samples_, self.n_samples_)
            else:
                bootstrap_indices = np.arange(self.n_samples_)

            weights = sample_weight * np.bincount(bootstrap_indices, minlength=self.n_samples_)
            self.y_weights_[i] = weights / est.tree_.weighted_n_node_samples[self.y_train_leaves_[i]]

        self.y_train_leaves_[self.y_weights_ == 0] = -1

        for i, est in enumerate(self.estimators_):
            if self.verbose:
                print(f"Sampling tree {i+1} of {self.n_estimators}")

            mask = est.y_weights_ > 0

            leaves = est.y_train_leaves_[mask]
            idx = np.arange(self.n_samples_, dtype=np.intp)[mask]

            unique_leaves = np.unique(leaves)

            sampled_values = np.full(len(unique_leaves), -np.inf, dtype=np.float32)
            _maximum_per_leaf(leaves, unique_leaves, est.y_train_[mask], idx, sampled_values, self.n_jobs)

            est.tree_.value[unique_leaves, 0, 0] = sampled_values

        return self

    def predict(self, X):
        """
        predition, based on the predict function of ForestRegressor
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        y_hat = np.zeros(X.shape[0], dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_accumulate_prediction)(e.predict, X, y_hat, lock)
            for e in self.estimators_)

        return y_hat
