# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=True

# Authors: Jasper Roebroek <roebroek.jasper@gmail.com>
# License: BSD 3 clause

"""
This module is inspired on the skgarden implementation of Forest Quantile Regression,
based on the following paper:

Nicolai Meinshausen, Quantile Regression Forests
http://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf
"""
from cython.parallel cimport prange
cimport openmp
cimport numpy as np
from numpy cimport ndarray

from sklearn_quantile.utils.weighted_quantile cimport _weighted_quantile_presorted_1D, \
    _weighted_quantile_unchecked_1D, Interpolation

from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.lib.function_base import _quantile_is_valid
import threading
import joblib
from joblib import Parallel

from sklearn.ensemble._forest import ForestRegressor, _accumulate_prediction, _generate_sample_indices
from sklearn.ensemble._base import _partition_estimators
from sklearn.utils.fixes import delayed
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.utils import check_array, check_X_y, check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_pinball_loss
from sklearn_quantile.base import QuantileRegressorMixin

ctypedef np.npy_intp SIZE_t              # Type for indices and counters


__all__ = ["RandomForestQuantileRegressor", "ExtraTreesQuantileRegressor", "SampleRandomForestQuantileRegressor",
           "SampleExtraTreesQuantileRegressor"]


cpdef void _quantile_forest_predict(SIZE_t[:, ::1] X_leaves,
                                    float[:, ::1] y_train,
                                    SIZE_t[:, ::1] y_train_leaves,
                                    float[:, ::1] y_weights,
                                    float[::1] q,
                                    float[:, :, ::1] quantiles,
                                    SIZE_t start,
                                    SIZE_t stop):
    """
    X_leaves : (n_estimators, n_test_samples)
    y_train : (n_samples, n_outputs)
    y_train_leaves : (n_estimators, n_samples)
    y_weights : (n_estimators, n_samples)
    q : (n_q)
    quantiles : output array (n_q, n_test_samples, n_outputs)
    start, stop : indices to break up computation across threads (used in range)
    """
    # todo; this does not compile with function cdef, only with cpdef

    cdef:
        int n_estimators = X_leaves.shape[0]
        int n_outputs = y_train.shape[1]
        int n_q = q.shape[0]
        int n_samples = y_train.shape[0]
        int n_test_samples = X_leaves.shape[1]

        int i, j, e, o, count_samples
        float curr_weight
        bint sorted = y_train.shape[1] == 1

        float[::1] x_weights = np.empty(n_samples, dtype=np.float32)
        float[:, ::1] x_a = np.empty((n_samples, n_outputs), dtype=np.float32)

    with nogil:
        for i in range(start, stop):
            count_samples = 0
            for j in range(n_samples):
                curr_weight = 0
                for e in range(n_estimators):
                    if X_leaves[e, i] == y_train_leaves[e, j]:
                        curr_weight = curr_weight + y_weights[e, j]
                if curr_weight > 0:
                    x_weights[count_samples] = curr_weight
                    x_a[count_samples] = y_train[j]
                    count_samples = count_samples + 1
            if sorted:
                _weighted_quantile_presorted_1D(x_a[:count_samples, 0],
                                                q, x_weights[:count_samples],
                                                quantiles[:, i, 0], Interpolation.linear)
            else:
                for o in range(n_outputs):
                    _weighted_quantile_unchecked_1D(x_a[:count_samples, o], q, x_weights[:count_samples],
                                                    quantiles[:, i, o], Interpolation.linear)


cdef void _weighted_random_sample(SIZE_t[::1] leaves,
                                  SIZE_t[::1] unique_leaves,
                                  float[::1] weights,
                                  SIZE_t[::1] idx,
                                  double[::1] random_numbers,
                                  SIZE_t[::1] sampled_idx,
                                  SIZE_t n_jobs):
    """
    Random sample for each unique leaf

    Parameters
    ----------
    leaves : array, shape = (n_samples)
        Leaves of a Regression tree, corresponding to weights and indices (idx)
    unique_leaves : array, shape = (n_unique_leaves)

    weights : array, shape = (n_samples)
        Weights for each observation. They need to sum up to 1 per unique leaf.
    idx : array, shape = (n_samples)
        Indices of original observations. The output will drawn from this.
    random numbers : array, shape = (n_unique_leaves)
    sampled_idx : shape = (n_unique_leaves)
    n_jobs : number of threads, similar to joblib
    """
    cdef:
        long c_leaf
        float p, r
        int i, j

        int n_unique_leaves = unique_leaves.shape[0]
        int n_samples = weights.shape[0]
        int num_threads = joblib.effective_n_jobs(n_jobs)

    for i in prange(n_unique_leaves, nogil=True, num_threads=num_threads):
        p = 0
        r = random_numbers[i]
        c_leaf = unique_leaves[i]

        for j in range(n_samples):
            if leaves[j] == c_leaf:
                p = p + weights[j]
                if p > r:
                    sampled_idx[i] = idx[j]
                    break


def _accumulate_prediction(predict, X, i, out, lock):
    """
    Adapted from sklearn.ensemble._forest
    """
    prediction = predict(X, check_input=False)
    with lock:
        if out.shape[1] == 1:
            out[:, 0, i] = prediction
        else:
            out[..., i] = prediction


class BaseForestQuantileRegressor(QuantileRegressorMixin, ForestRegressor, metaclass=ABCMeta):
    """
    A baseclass forest regressor, providing conditional quantile estimates.
    The baseclass implements a `fit` method that is required to overwrite.
    It stores information on the generated weights from the bootstrap process
    in the base estimators.
    """
    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """
        Build a forest from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix, shape = (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            The target values

        sample_weight : array-like, shape = (n_samples) or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        Returns
        -------
        self : object
            Returns self.
        """
        # apply method requires X to be of dtype np.float32
        # multi-output should likely work, but tests need to be written first.
        #   known case of error: maximum regressor
        X, y = check_X_y(
            X, y, accept_sparse="csc", dtype=np.float32, multi_output=False)
        super(BaseForestQuantileRegressor, self).fit(X, y, sample_weight=sample_weight)

        self.n_samples_ = len(y)
        self.y_train_ = y.reshape((-1, self.n_outputs_)).astype(np.float32)
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

            if self.bootstrap:
                bootstrap_indices = _generate_sample_indices(
                    est.random_state, self.n_samples_, self.n_samples_)
            else:
                bootstrap_indices = np.arange(self.n_samples_)

            weights = sample_weight * np.bincount(bootstrap_indices, minlength=self.n_samples_)
            est.y_weights_[:] = weights / est.tree_.weighted_n_node_samples[est.y_train_leaves_]

        self.y_train_leaves_[self.y_weights_ == 0] = -1

    @abstractmethod
    def predict(self, X):
        """
        Predict conditional quantiles for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = (n_quantiles, n_samples, n_outputs)
            return y such that F(Y=y | x) = quantile. If n_quantiles is 1, the array is reduced to
            (n_samples, n_outputs) and if n_outputs is 1, the array is reduced to (n_samples)
        """
        raise NotImplementedError


class ForestQuantileRegressor(BaseForestQuantileRegressor, metaclass=ABCMeta):
    """
    Base class for forest quantile regressors (both random forest and extree trees),
    fitting and predicting according to the original paper by Meinshausen (2006).
        http://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf
    """
    def fit(self, X, y, sample_weight=None):
        super(ForestQuantileRegressor, self).fit(X, y, sample_weight)

        if self.n_outputs_ == 1:
            sort_ind = np.argsort(y)
            self.y_train_[:] = self.y_train_[sort_ind]
            self.y_weights_[:] = self.y_weights_[:, sort_ind]
            self.y_train_leaves_[:] = self.y_train_leaves_[:, sort_ind]
            self.y_sorted_ = True
        else:
            self.y_sorted_ = False

        return self

    def predict(self, X):
        check_is_fitted(self)
        q = self.validate_quantiles()

        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # apply method requires X to be of dtype np.float32
        X = check_array(X, dtype=np.float32, accept_sparse="csc")
        X_leaves = self.apply(X).T
        n_test_samples = X.shape[0]

        chunks = np.full(n_jobs, n_test_samples//n_jobs)
        chunks[:n_test_samples % n_jobs] +=1
        chunks = np.cumsum(np.insert(chunks, 0, 0))

        quantiles = np.empty((q.size, n_test_samples, self.n_outputs_), dtype=np.float32)

        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_quantile_forest_predict)(X_leaves, self.y_train_, self.y_train_leaves_, self.y_weights_, q,
                                              quantiles, chunks[i], chunks[i+1])
            for i in range(n_jobs))

        if q.size == 1:
            quantiles = quantiles[0]
        if self.n_outputs_ == 1:
            quantiles = quantiles[..., 0]

        return quantiles


class SampleForestQuantileRegressor(BaseForestQuantileRegressor, metaclass=ABCMeta):
    """
    Base class for forest quantile regressors (both random forest and extree trees),
    fitting and predicting according to the approximation method chosen in the R package
    quantregForest.
    """
    def fit(self, X, y, sample_weight=None):
        super(SampleForestQuantileRegressor, self).fit(X, y, sample_weight=sample_weight)

        for i, est in enumerate(self.estimators_):
            if self.verbose:
                print(f"Sampling tree {i+1} of {self.n_estimators}")

            mask = est.y_weights_ > 0

            leaves = est.y_train_leaves_[mask]
            idx = np.arange(self.n_samples_, dtype=np.intp)[mask]

            unique_leaves = np.unique(leaves)

            random_instance = check_random_state(est.random_state)
            random_numbers = random_instance.rand(len(unique_leaves))

            sampled_idx = np.empty(len(unique_leaves), dtype=np.intp)
            n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

            _weighted_random_sample(leaves, unique_leaves, est.y_weights_[mask], idx, random_numbers, sampled_idx,
                                    n_jobs)

            est.tree_.value[unique_leaves, :, 0] = self.y_train_[sampled_idx]

        return self

    def predict(self, X):
        check_is_fitted(self)
        q = self.validate_quantiles()

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # apply method requires X to be of dtype np.float32
        X = check_array(X, dtype=np.float32, accept_sparse="csc")

        predictions = np.empty((len(X), self.n_outputs_, self.n_estimators))

        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_accumulate_prediction)(est.predict, X, i, predictions, lock)
            for i, est in enumerate(self.estimators_))

        quantiles = np.quantile(predictions, q=q, axis=-1)
        if q.size == 1:
            quantiles = quantiles[0]
        if self.n_outputs_ == 1:
            quantiles = quantiles[..., 0]

        return quantiles


### The following classes combine the implementation and base_estimator and can be used for modelling
class RandomForestQuantileRegressor(ForestQuantileRegressor):
    """
    A random forest regressor providing quantile estimates.

    Note that this implementation is rather slow for large datasets. Above 10000 samples
    it is recommended to use func:`sklearn_quantile.SampleRandomForestQuantileRegressor`,
    which is a model approximating the true conditional quantile.

    Parameters
    ----------
    q : float or array-like, optional
        Quantiles used for prediction (values ranging from 0 to 1)

    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.
        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
        .. versionchanged:: 0.18
           Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
        .. versionchanged:: 0.18
           Added float values for percentages.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    oob_score : bool, optional (default=False)
        whether to use out-of-bag samples to estimate
        the R^2 on unseen data.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_prediction_ : array of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training set.

    References
    ----------
    .. [1] Nicolai Meinshausen, Quantile Regression Forests
        http://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf
    """
    def __init__(self,
                 q=None,
                 n_estimators=10,
                 criterion='mse',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(BaseForestQuantileRegressor, self).__init__(
            base_estimator=DecisionTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.q = q


class ExtraTreesQuantileRegressor(ForestQuantileRegressor):
    """
    A extra trees regressor providing quantile estimates.

    Note that this implementation is rather slow for large datasets. Above 10000 samples
    it is recommended to use func:`sklearn_quantile.SampleExtraTreesQuantileRegressor`,
    which is a model approximating the true conditional quantile.

    Parameters
    ----------
    q : float or array-like, optional
        Quantiles used for prediction (values ranging from 0 to 1)

    n_estimators : int, default=100
        The number of trees in the forest.

        .. versionchanged:: 0.22
           The default value of ``n_estimators`` changed from 10 to 100
           in 0.22.

    criterion : {"squared_error", "absolute_error"}, default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion, and "absolute_error"
        for the mean absolute error.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

        .. deprecated:: 1.0
            Criterion "mse" was deprecated in v1.0 and will be removed in
            version 1.2. Use `criterion="squared_error"` which is equivalent.

        .. deprecated:: 1.0
            Criterion "mae" was deprecated in v1.0 and will be removed in
            version 1.2. Use `criterion="absolute_error"` which is equivalent.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : {"sqrt", "log2", None}, int or float, default=1.0
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None or 1.0, then `max_features=n_features`.

        .. note::
            The default of 1.0 is equivalent to bagged trees and more
            randomness can be achieved by setting smaller values, e.g. 0.3.

        .. versionchanged:: 1.1
            The default of `max_features` changed from `"auto"` to 1.0.

        .. deprecated:: 1.1
            The `"auto"` option was deprecated in 1.1 and will be removed
            in 1.3.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    bootstrap : bool, default=False
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
        Only available if bootstrap=True.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.

    random_state : int, RandomState instance or None, default=None
        Controls 3 sources of randomness:

        - the bootstrapping of the samples used when building trees
          (if ``bootstrap=True``)
        - the sampling of the features to consider when looking for the best
          split at each node (if ``max_features < n_features``)
        - the draw of the splits for each of the `max_features`

        See :term:`Glossary <random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0.0, 1.0]`.

        .. versionadded:: 0.22

    Attributes
    ----------
    base_estimator_ : ExtraTreeRegressor
        The child estimator template used to create the collection of fitted
        sub-estimators.

    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    n_features_ : int
        The number of features.

        .. deprecated:: 1.0
            Attribute `n_features_` was deprecated in version 1.0 and will be
            removed in 1.2. Use `n_features_in_` instead.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_outputs_ : int
        The number of outputs.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.

    oob_prediction_ : ndarray of shape (n_samples,) or (n_samples, n_outputs)
        Prediction computed with out-of-bag estimate on the training set.
        This attribute exists only when ``oob_score`` is True.

    See Also
    --------
    ExtraTreesClassifier : An extra-trees classifier with random splits.
    RandomForestClassifier : A random forest classifier with optimal splits.
    RandomForestRegressor : Ensemble regressor using trees with optimal splits.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------
    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import ExtraTreesRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=0)
    >>> reg = ExtraTreesRegressor(n_estimators=100, random_state=0).fit(
    ...    X_train, y_train)
    >>> reg.score(X_test, y_test)
    0.2727...
    """
    def __init__(
        self,
        n_estimators=100,
        q=None,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        super().__init__(
            base_estimator=ExtraTreeRegressor(),
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
        self.q = q


class SampleRandomForestQuantileRegressor(SampleForestQuantileRegressor):
    """
    An approximation random forest regressor providing quantile estimates.

    Note that this implementation is a fast approximation of a Random Forest
    Quanatile Regressor. It is useful in cases where performance is important.
    For mathematical accuracy use :func:`sklearn_quantile.RandomForestQuantileRegressor`.

    Parameters
    ----------
    q : float or array-like, optional
        Quantiles used for prediction (values ranging from 0 to 1)

    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.
        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
        .. versionchanged:: 0.18
           Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
        .. versionchanged:: 0.18
           Added float values for percentages.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    oob_score : bool, optional (default=False)
        whether to use out-of-bag samples to estimate
        the R^2 on unseen data.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_prediction_ : array of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training set.

    References
    ----------
    .. [1] Nicolai Meinshausen, Quantile Regression Forests
        http://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf
    """
    def __init__(self,
                 q=None,
                 n_estimators=10,
                 criterion='mse',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(BaseForestQuantileRegressor, self).__init__(
            base_estimator=DecisionTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.q = q


class SampleExtraTreesQuantileRegressor(SampleForestQuantileRegressor):
    """
    An approximation extra trees regressor providing quantile estimates.

    Note that this implementation is a fast approximation of a Extra Trees
    Quanatile Regressor. It is useful in cases where performance is important.
    For mathematical accuracy use :func:`sklearn_quantile.ExtraTreesQuantileRegressor`.

    Parameters
    ----------
    q : float or array-like, optional
        Quantiles used for prediction (values ranging from 0 to 1)

    n_estimators : int, default=100
        The number of trees in the forest.

        .. versionchanged:: 0.22
           The default value of ``n_estimators`` changed from 10 to 100
           in 0.22.

    criterion : {"squared_error", "absolute_error"}, default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion, and "absolute_error"
        for the mean absolute error.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

        .. deprecated:: 1.0
            Criterion "mse" was deprecated in v1.0 and will be removed in
            version 1.2. Use `criterion="squared_error"` which is equivalent.

        .. deprecated:: 1.0
            Criterion "mae" was deprecated in v1.0 and will be removed in
            version 1.2. Use `criterion="absolute_error"` which is equivalent.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : {"sqrt", "log2", None}, int or float, default=1.0
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None or 1.0, then `max_features=n_features`.

        .. note::
            The default of 1.0 is equivalent to bagged trees and more
            randomness can be achieved by setting smaller values, e.g. 0.3.

        .. versionchanged:: 1.1
            The default of `max_features` changed from `"auto"` to 1.0.

        .. deprecated:: 1.1
            The `"auto"` option was deprecated in 1.1 and will be removed
            in 1.3.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    bootstrap : bool, default=False
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
        Only available if bootstrap=True.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.

    random_state : int, RandomState instance or None, default=None
        Controls 3 sources of randomness:

        - the bootstrapping of the samples used when building trees
          (if ``bootstrap=True``)
        - the sampling of the features to consider when looking for the best
          split at each node (if ``max_features < n_features``)
        - the draw of the splits for each of the `max_features`

        See :term:`Glossary <random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0.0, 1.0]`.

        .. versionadded:: 0.22

    Attributes
    ----------
    base_estimator_ : ExtraTreeRegressor
        The child estimator template used to create the collection of fitted
        sub-estimators.

    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    n_features_ : int
        The number of features.

        .. deprecated:: 1.0
            Attribute `n_features_` was deprecated in version 1.0 and will be
            removed in 1.2. Use `n_features_in_` instead.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_outputs_ : int
        The number of outputs.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.

    oob_prediction_ : ndarray of shape (n_samples,) or (n_samples, n_outputs)
        Prediction computed with out-of-bag estimate on the training set.
        This attribute exists only when ``oob_score`` is True.

    See Also
    --------
    ExtraTreesClassifier : An extra-trees classifier with random splits.
    RandomForestClassifier : A random forest classifier with optimal splits.
    RandomForestRegressor : Ensemble regressor using trees with optimal splits.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------
    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import ExtraTreesRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=0)
    >>> reg = ExtraTreesRegressor(n_estimators=100, random_state=0).fit(
    ...    X_train, y_train)
    >>> reg.score(X_test, y_test)
    0.2727...
    """
    def __init__(
        self,
        n_estimators=100,
        q=None,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        super().__init__(
            base_estimator=ExtraTreeRegressor(),
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
        self.q = q
