from functools import partial

import numpy as np
import pytest
from numpy.ma.testutils import assert_array_almost_equal
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn_quantile.utils.weighted_quantile import WeightedQuantileCalculator

from sklearn_quantile import KNeighborsQuantileRegressor

np_quantile = partial(np.quantile, method="interpolated_inverted_cdf")

X, y = fetch_california_housing(return_X_y=True)
X = X[:500]
y = y[:500]


@pytest.fixture
def weighted_quantile():
    wqc = WeightedQuantileCalculator()
    weighted_quantile = wqc.py_weighted_quantile
    return weighted_quantile


def test_return_shape():
    m = KNeighborsQuantileRegressor(q=0.5)

    m.fit(X, y)
    assert m.predict(X).shape == (500,)

    m.fit(X, np.vstack([y, y]).T)
    assert m.predict(X).shape == (500, 2)

    m.q = [0.5, 0.5]

    m.fit(X, y)
    assert m.predict(X).shape == (2, 500)

    m.fit(X, np.vstack([y, y]).T)
    assert m.predict(X).shape == (2, 500, 2)


@pytest.mark.parametrize("q", [0, 0.1, 0.5, 0.9, 1])
def test_knn_equivalence_n_1(q):
    """With only one neighbour, it shouldn't matter which quantile you are asking for"""
    qknn = KNeighborsQuantileRegressor(n_neighbors=1, q=q)
    knn = KNeighborsRegressor(n_neighbors=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.5, random_state=0
    )
    qknn.fit(X_train, y_train)
    knn.fit(X_train, y_train)

    assert_array_almost_equal(qknn.predict(X), knn.predict(X))


@pytest.mark.parametrize("q", [0, 0.1, 0.9, 1])
def test_knn_difference_n_2(q):
    """With two neighbours, the outcome between mean and quantiles start to matter"""
    qknn = KNeighborsQuantileRegressor(n_neighbors=2, q=q)
    knn = KNeighborsRegressor(n_neighbors=2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.5, random_state=0
    )
    qknn.fit(X_train, y_train)
    knn.fit(X_train, y_train)

    assert not np.array_equal(qknn.predict(X), knn.predict(X))


@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
def test_quantile_values_equal_weights(q):
    """with uniform weights and as many neighbours as values, the quantile should be same as the
    quantile on the data"""
    m = KNeighborsQuantileRegressor(n_neighbors=5, q=q)
    X = np.asarray([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.asarray([10, 12, 98, 7, -1])

    m.fit(X, y)

    assert np.isclose(m.predict([[6]])[0], np_quantile(y, q))


def test_quantile_values_distance_weights():
    """with non_uniform weights, this should not be the same"""

    X = np.asarray([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.asarray([10, 12, 98, 7, -1])

    m = KNeighborsQuantileRegressor(n_neighbors=5, q=0.5, weights="distance")
    m.fit(X, y)

    # Value should be closer to the observation -1
    assert m.predict([[6]]).item() < np_quantile(y, q=0.5).item()


@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
def test_knn_weights_ones(q, weighted_quantile):
    r = np.random.RandomState(0)
    X = r.rand(10, 2)
    y = r.rand(10)

    m = KNeighborsQuantileRegressor(
        n_neighbors=5, q=q, weights=lambda x: np.ones_like(x)
    )

    m.fit(X, y)
    dist, ind = m.kneighbors(X)
    pred = [weighted_quantile(a=y[ind][i], weights=np.ones(5), q=q).item() for i in range(10)]

    assert_array_almost_equal(m.predict(X), pred)


@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
def test_knn_weights(q, weighted_quantile):
    r = np.random.RandomState(0)
    X = r.rand(10, 2)
    y = r.rand(10)

    m1 = KNeighborsQuantileRegressor(
        n_neighbors=5, q=q, weights=lambda x: np.ones_like(x)
    )
    m2 = KNeighborsQuantileRegressor(n_neighbors=5, q=q, weights="uniform")
    m3 = KNeighborsQuantileRegressor(n_neighbors=5, q=q, weights="distance")

    m1.fit(X, y)
    m2.fit(X, y)
    m3.fit(X, y)

    assert_array_almost_equal(m1.predict(X), m2.predict(X))
    assert not np.array_equal(m1.predict(X), m3.predict(X))

    # if weights are based on distance, and X_test is the same as X_train, the predictions are the same
    # as the training data
    assert_array_almost_equal(m3.predict(X), y)
