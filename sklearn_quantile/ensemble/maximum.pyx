# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Jasper Roebroek <roebroek.jasper@gmail.com>
# License: BSD 3 clause

"""
Module providing the special case for quantile regression: predicting the maximum.
It only has a single implementation called RandomForestMaximumRegressor, which has the
same parameters as the regular RandomForestRegressor from scikit-learn
"""
import threading

import numpy as np
cimport numpy as cnp
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.parallel import delayed, Parallel
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble._base import _partition_estimators
from sklearn.ensemble._forest import ForestRegressor

from sklearn_quantile.ensemble.quantile import BaseForestQuantileRegressor


ctypedef cnp.intp_t SSIZE_t  # Type for indices and counters


__all__ = ["RandomForestMaximumRegressor"]


cdef void _maximum_per_leaf(SSIZE_t[::1] leaves,
                            float[::1] values,
                            float[::1] tree_values) nogil:

    """
    Store the highest value found for each leaf in the tree_values array.
    
    Parameters
    ----------
    leaves : array, shape = (n_samples)
        Leaves of a Regression tree, corresponding to locations in the tree_values array.
    values : array, shape = (n_samples)
    tree_values: array, shape = (n_nodes)
        Initialised with -np.inf. Filled at leaf nodes with highest value
    """
    cdef:
        int i
        ssize_t c_leaf
        int n_samples = leaves.shape[0]

    with nogil:
        for i in range(n_samples):
            c_leaf = leaves[i]
            if c_leaf == -1:
                continue
            if values[i] > tree_values[c_leaf]:
                tree_values[c_leaf] = values[i]


def _fit_each_tree(est):
    values = np.full(est.tree_.node_count, dtype=np.float32, fill_value=-np.inf)

    _maximum_per_leaf(leaves=est.y_train_leaves_,
                      values=est.y_train_[:, 0],
                      tree_values=values)

    est.tree_.value[:, 0, 0] = values


def _accumulate_prediction(predict, X, out, lock):
    """
    This is a utility function for joblib's Parallel. Based on the version in sklearn.ensemble._forest
    """
    prediction = predict(X, check_input=False)
    with lock:
        out[:] = np.maximum(prediction, out)


class RandomForestMaximumRegressor(BaseForestQuantileRegressor):
    """
    A random forest regressor predicting conditional maxima

    Implementation is equivalent to Random Forest Quantile Regressor,
    but calculation is much faster. For other quantiles revert to the
    original predictor.
    """
    _parameter_constraints: dict = {
        **ForestRegressor._parameter_constraints,
        **DecisionTreeRegressor._parameter_constraints,
    }
    _parameter_constraints.pop("splitter")

    def __init__(
            self,
            n_estimators=100,
            *,
            criterion="squared_error",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=1.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
            monotonic_cst=None,
    ):
        super().__init__(
            estimator=DecisionTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
                "monotonic_cst",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst
        self.q = 1

    def fit(self, X, y, sample_weight=None):
        super(RandomForestMaximumRegressor, self).fit(X, y, sample_weight)

        if self.verbose:
            print("Sampling maximum per tree")

        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_fit_each_tree)(e)
            for e in self.estimators_)

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

        y_hat = np.full(X.shape[0], dtype=np.float32, fill_value=-np.inf)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict, X, y_hat, lock)
            for e in self.estimators_)

        return y_hat
