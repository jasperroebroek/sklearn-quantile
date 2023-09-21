import pytest
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from sklearn_quantile import (
    RandomForestQuantileRegressor,
    ExtraTreesQuantileRegressor,
    SampleRandomForestQuantileRegressor,
    SampleExtraTreesQuantileRegressor,
    RandomForestMaximumRegressor
)

X, y = fetch_california_housing(return_X_y=True)
X = X[:500]
y = y[:500]


# todo; additional tests
#   - test if summed estimator.y_weights_ per leaf are equal to 1
#   - test if weights


@pytest.mark.parametrize("model", [RandomForestQuantileRegressor,
                                   ExtraTreesQuantileRegressor,
                                   SampleRandomForestQuantileRegressor,
                                   SampleExtraTreesQuantileRegressor,
                                   RandomForestMaximumRegressor])
def test_common_attributes(model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=0)

    m = model(random_state=0)
    m.fit(X_train, y_train)

    # If a sample is not present in a particular tree, that
    # corresponding leaf is marked as -1.
    assert_array_equal(
        np.vstack(np.where(m.y_train_leaves_ == -1)),
        np.vstack(np.where(m.y_weights_ == 0))
    )

    # Should sum up to number of leaf nodes.
    assert_array_almost_equal(
        np.sum(m.y_weights_, axis=1),
        [sum(tree.tree_.children_left == -1) for tree in m.estimators_],
        0
    )

    # Sharing memory between trees and forest
    assert np.all([np.may_share_memory(m.y_weights_, m.estimators_[i].y_weights_)
                   for i in range(m.n_estimators)])
    assert np.all([np.may_share_memory(m.y_train_leaves_, m.estimators_[i].y_train_leaves_)
                   for i in range(m.n_estimators)])

    # When bootstrapping is not active, no y_train_leaves should be -1
    m.set_params(bootstrap=False)
    m.fit(X_train, y_train)
    assert np.all(m.y_train_leaves_ != -1)


@pytest.mark.parametrize("model,base_model,passes", [(RandomForestQuantileRegressor, RandomForestRegressor, True),
                                                     (ExtraTreesQuantileRegressor, ExtraTreesRegressor, True),
                                                     (SampleRandomForestQuantileRegressor, RandomForestRegressor, False),
                                                     (SampleExtraTreesQuantileRegressor, ExtraTreesRegressor, False),
                                                     (RandomForestMaximumRegressor, RandomForestRegressor, False)])
def test_value_array_against_RF(model, base_model, passes):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=0)

    # Prevent splitting, so the arrays are forced to be different
    base_m = base_model(random_state=0, min_samples_split=y_train.size + 1)
    m = model(random_state=0, min_samples_split=y_train.size + 1)

    base_m.fit(X_train, y_train)
    m.fit(X_train, y_train)

    assert passes == np.all([
        np.array_equal(m.estimators_[i].tree_.value, base_m.estimators_[i].tree_.value)
        for i in range(m.n_estimators)])


def test_median_value_equal():
    """Median should very closely follow RandomForestQuantileRegression at q=0.5"""

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=0)

    # Prevent splitting, so the arrays are forced to be different
    rf = RandomForestRegressor(random_state=0, criterion='absolute_error')
    qrf = RandomForestQuantileRegressor(random_state=0, q=0.5)

    rf.fit(X_train, y_train)
    qrf.fit(X_train, y_train)

    assert np.allclose(rf.predict(X_test), qrf.predict(X_test), atol=0.4)


def test_median_value_not_equal():
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=0)

    # Prevent splitting, so the arrays are forced to be different
    rf = RandomForestRegressor(random_state=0, criterion='absolute_error')
    qrf = RandomForestQuantileRegressor(random_state=0, q=1)

    rf.fit(X_train, y_train)
    qrf.fit(X_train, y_train)

    assert ~np.allclose(rf.predict(X_test), qrf.predict(X_test), atol=0.4)



@pytest.mark.parametrize("n_jobs", [1, 2])
def test_maximum_regressor_for_equality(n_jobs):
    # maximum regressor should be equal to the standard implementation
    qrf = RandomForestQuantileRegressor(random_state=0, q=1, n_jobs=n_jobs)
    maximum_rf = RandomForestMaximumRegressor(random_state=0, n_jobs=n_jobs)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=0)

    qrf.fit(X_train, y_train)
    maximum_rf.fit(X_train, y_train)

    qrf_prediction = qrf.predict(X_test)
    maximum_rf_prediction = maximum_rf.predict(X_test)

    assert_array_almost_equal(qrf_prediction, maximum_rf_prediction)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_maximum_regressor(n_jobs):
    x = np.vstack((np.arange(40), np.hstack((np.arange(20), np.arange(20))))).T
    y = np.arange(40)

    rf = RandomForestMaximumRegressor(n_estimators=1, max_depth=2, bootstrap=False, n_jobs=n_jobs)
    rf.fit(x, y)
    assert_array_equal(rf.predict(x[[0, 10, 20, 30]]), [9, 19, 29, 39])


@pytest.mark.parametrize('q', [0, 0.1, 0.5, 0.9, 1])
@pytest.mark.parametrize('model', [RandomForestQuantileRegressor,
                                   ExtraTreesQuantileRegressor])
def test_values(model,q):
    m = model(n_estimators=1, bootstrap=False, min_samples_split=y.size + 1, q=q)
    m.fit(X, y)

    assert np.isclose(m.predict(X[0:1]), np.quantile(y, q), rtol=0.01)


@pytest.mark.parametrize('q', [(0.5), (0.5, 0.5), (0.5, 0.5, 0.5)])
@pytest.mark.parametrize("model", [RandomForestQuantileRegressor,
                                   ExtraTreesQuantileRegressor,
                                   SampleRandomForestQuantileRegressor,
                                   SampleExtraTreesQuantileRegressor])
def test_output_shape(model, q):
    q = np.asarray(q)
    m = model(q=q)
    m.fit(X, y)

    if q.size == 1:
        assert m.predict(X).shape == (500,)
    else:
        assert m.predict(X).shape == (q.size, 500)


@pytest.mark.parametrize('n_jobs', [1, 2])
@pytest.mark.parametrize('q', [0, 0.1, 0.5, 0.9, 1])
@pytest.mark.parametrize("model,base_model", [(SampleRandomForestQuantileRegressor, RandomForestQuantileRegressor),
                                              (SampleExtraTreesQuantileRegressor, ExtraTreesQuantileRegressor)])
def test_sample_implementation_precision(model, base_model, q, n_jobs):
    """Values between exact and sample implementation are not guaranteed to be the same. Nevertheless, a
    very high correlation between them can be assumed. The exact threshold is not important here, but should
    not be changed too much"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=0)

    m = model(q=q, random_state=0, n_jobs=n_jobs)
    base_m = base_model(q=q, random_state=0, n_jobs=n_jobs)

    m.fit(X_train, y_train)
    base_m.fit(X_train, y_train)

    assert r2_score(m.predict(X_test), base_m.predict(X_test)) > 0.9
