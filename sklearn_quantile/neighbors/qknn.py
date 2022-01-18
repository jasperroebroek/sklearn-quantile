from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.neighbors._base import _get_weights
import numpy as np
from numpy.lib.function_base import _quantile_is_valid

from sklearn_quantile.utils import weighted_quantile


class KNeighborsQuantileRegressor(KNeighborsRegressor):
    """Quantile regression based on k-nearest neighbors.

    The target is predicted by local interpolation of the targets
    associated of the nearest neighbors in the training set.

    Read more in the :ref:`User Guide <regression>`.

    .. versionadded:: 0.9

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    weights : {'uniform', 'distance'} or callable, default='uniform'
        Weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

        Uniform weights are used by default.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    p : int, default=2
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric : str or callable, default='minkowski'
        The distance metric to use for the tree.  The default metric is
        minkowski, and with p=2 is equivalent to the standard Euclidean
        metric. See the documentation of :class:`DistanceMetric` for a
        list of available metrics.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a :term:`sparse graph`,
        in which case only "nonzero" elements may be considered neighbors.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        Doesn't affect :meth:`fit` method.

    quantiles : array-like, optional
        Value ranging from 0 to 1

    Attributes
    ----------
    effective_metric_ : str or callable
        The distance metric to use. It will be same as the `metric` parameter
        or a synonym of it, e.g. 'euclidean' if the `metric` parameter set to
        'minkowski' and `p` parameter set to 2.

    effective_metric_params_ : dict
        Additional keyword arguments for the metric function. For most metrics
        will be same with `metric_params` parameter, but may also contain the
        `p` parameter value if the `effective_metric_` attribute is set to
        'minkowski'.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_samples_fit_ : int
        Number of samples in the fitted data.

    See Also
    --------
    NearestNeighbors : Unsupervised learner for implementing neighbor searches.
    RadiusNeighborsRegressor : Regression based on neighbors within a fixed radius.
    KNeighborsClassifier : Classifier implementing the k-nearest neighbors vote.
    RadiusNeighborsClassifier : Classifier implementing
        a vote among neighbors within a given radius.

    Notes
    -----
    See :ref:`Nearest Neighbors <neighbors>` in the online documentation
    for a discussion of the choice of ``algorithm`` and ``leaf_size``.

    .. warning::

       Regarding the Nearest Neighbors algorithms, if it is found that two
       neighbors, neighbor `k+1` and `k`, have identical distances but
       different labels, the results will depend on the ordering of the
       training data.

    https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

    """
    def __init__(self, quantiles=None, **kwargs):
        super(KNeighborsQuantileRegressor, self).__init__(**kwargs)
        self.quantiles = quantiles

    def validate_quantiles(self):
        if self.quantiles is None:
            raise AttributeError("Quantiles are not set. Please provide them with `model.quantiles = quantiles`")
        q = np.asarray(self.quantiles, dtype=np.float32)
        q = np.atleast_1d(q)
        if q.ndim > 2:
            raise ValueError("q must be a scalar or 1D")

        if not _quantile_is_valid(q):
            raise ValueError("Quantiles must be in the range [0, 1]")

        return q

    def score(self, X, y):
        q = self.validate_quantiles()
        y_pred = self.predict(X)
        losses = np.empty(q.size)
        if q.size == 1:
            return mean_pinball_loss(y, y_pred, alpha=q.item())
        else:
            for i in range(q.size):
                losses[i] = mean_pinball_loss(y, y_pred[i], alpha=q[i])
            return np.mean(losses)

    def predict(self, X):
        """
        Predict conditional quantile `q` of the nearest neighbours.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs), dtype=float
            Target values.
        """
        q = self.validate_quantiles()
        X = self._validate_data(X, accept_sparse='csr', reset=False)

        neigh_dist, neigh_ind = self.kneighbors(X)

        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        a = _y[neigh_ind]
        if weights is not None:
            weights = np.broadcast_to(weights[:, :, np.newaxis], a.shape)

        # this falls back on np.quantile if weights is None
        y_pred = weighted_quantile(a, q, weights, axis=1)

        if self._y.ndim == 1:
            y_pred = y_pred[..., 0]

        if q.size == 1:
            y_pred = y_pred[0]

        return y_pred
