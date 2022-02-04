import pytest
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import fetch_california_housing
import numpy as np
from numpy.testing import assert_array_equal

from sklearn_quantile import KNeighborsQuantileRegressor

X, y = fetch_california_housing(return_X_y=True)
X = X[:500]
y = y[:500]


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

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=0)
    qknn.fit(X_train, y_train)
    knn.fit(X_train, y_train)

    assert_array_equal(qknn.predict(X), knn.predict(X))


@pytest.mark.parametrize("q", [0, 0.1, 0.9, 1])
def test_knn_difference_n_2(q):
    """With two neighbours, the outcome between mean and quantiles start to matter"""
    qknn = KNeighborsQuantileRegressor(n_neighbors=2, q=q)
    knn = KNeighborsRegressor(n_neighbors=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=0)
    qknn.fit(X_train, y_train)
    knn.fit(X_train, y_train)

    assert not np.array_equal(qknn.predict(X), knn.predict(X))


@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
def test_quantile_values(q):
    """with uniform weights and as many neighbours as values, the quantile should be same as the
    quantile on the data"""
    m = KNeighborsQuantileRegressor(n_neighbors=5, q=q)
    X = np.asarray([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.asarray([10, 12, 98, 7, -1])

    m.fit(X, y)

    assert np.isclose(m.predict([[6]])[0], np.quantile(y, q))

    # with non_uniform weights, this should not be the same
    m = KNeighborsQuantileRegressor(n_neighbors=5, q=q, weights='distance')
    m.fit(X, y)

    assert not np.isclose(m.predict([[6]])[0], np.quantile(y, q))
