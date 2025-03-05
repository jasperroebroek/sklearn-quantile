from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from sklearn_quantile.utils.weighted_quantile import (
    WeightedQuantileCalculator,
    weighted_quantile as py_weighted_quantile,
)

np_quantile = partial(np.quantile, method="interpolated_inverted_cdf")


@pytest.fixture
def weighted_quantile():
    wqc = WeightedQuantileCalculator()
    weighted_quantile = wqc.py_weighted_quantile
    return weighted_quantile


def test_quantile_equal_weights_overlapping(weighted_quantile):
    rng = np.random.RandomState(0)
    x = rng.randn(10)
    weights = 0.1 * np.ones(10)
    q = np.arange(0.1, 1.0, 0.1)

    # since weights are equal, quantiles overlap with the values
    actual = weighted_quantile(a=x, weights=weights, q=q)
    assert_array_almost_equal(actual, np.sort(x)[:-1])


def test_quantile_equal_weights_not_overlapping(weighted_quantile):
    rng = np.random.RandomState(0)
    x = rng.randn(10)

    weights = 0.1 * np.ones(10)
    q = np.arange(0.05, 1.05, 0.1)

    actual = weighted_quantile(a=x, weights=weights, q=q)

    sorted_x = np.sort(x)
    expected = np.hstack(((sorted_x[0],), 0.5 * (sorted_x[1:] + sorted_x[:-1])))

    assert_array_almost_equal(actual, expected)


def test_weighted_quantile_toy_data(weighted_quantile):
    x = [1, 2, 3]
    weights = [1, 4, 5]

    assert_equal(weighted_quantile(a=x, weights=weights, q=0.0), 1)
    assert_equal(weighted_quantile(a=x, weights=weights, q=1.0), 3)

    assert_equal(weighted_quantile(a=x, weights=weights, q=0.05), 1)
    assert_almost_equal(weighted_quantile(a=x, weights=weights, q=0.30), 1.5)
    assert_equal(weighted_quantile(a=x, weights=weights, q=0.75), 2.5)
    assert_almost_equal(weighted_quantile(a=x, weights=weights, q=0.50), 2)


@pytest.mark.parametrize("q", [0, 0.1, 0.5, 0.9, 1])
def test_quantile_toy_data(q, weighted_quantile):
    x = [1, 2, 3]
    assert_almost_equal(
        weighted_quantile(a=x, weights=None, q=q), np_quantile(a=x, q=q), decimal=3
    )


def test_one_sample_per_quantile(weighted_quantile):
    x = np.asarray([1])
    weights = None

    assert_equal(weighted_quantile(a=x, weights=weights, q=0), 1)
    assert_equal(py_weighted_quantile(x, weights=weights, q=0), 1)

    weights = np.asarray([1])
    assert_equal(weighted_quantile(a=x, weights=weights, q=0), 1)
    assert_equal(py_weighted_quantile(x, weights=weights, q=0), 1)


@pytest.mark.parametrize("q", [0, 0.1, 0.5, 0.9, 1])
def test_zero_weights(q, weighted_quantile):
    x = [1, 2, 3, 4, 5]
    w = [0, 0, 0, 0.1, 0.1]

    assert_equal(
        weighted_quantile(a=x, weights=w, q=q),
        weighted_quantile(a=[4, 5], weights=[0.1, 0.1], q=q),
    )


@pytest.mark.parametrize("keepdims", [True, False])
def test_return_shapes_no_weights(keepdims):
    rng = np.random.RandomState(0)
    x = rng.randn(100, 10, 20)

    assert (
        py_weighted_quantile(x, 0.5, weights=None, axis=1, keepdims=keepdims).shape
        == np.quantile(x, 0.5, axis=1, keepdims=keepdims).shape
    )
    assert (
        py_weighted_quantile(
            x, (0.5, 0.5), weights=None, axis=1, keepdims=keepdims
        ).shape
        == np.quantile(x, (0.5, 0.5), axis=1, keepdims=keepdims).shape
    )


@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_return_shapes_with_axis(keepdims, axis):
    rng = np.random.RandomState(0)
    x = rng.randn(100, 10, 20)
    weights = 0.01 * np.ones_like(x)

    # shape should be the same as the output of np.quantile. Without weights it is actually the same calculation
    assert (
        py_weighted_quantile(x, 0.5, weights, axis=axis, keepdims=keepdims).shape
        == np.quantile(x, 0.5, axis=axis, keepdims=keepdims).shape
    )
    assert (
        py_weighted_quantile(x, (0.5, 0.8), weights, axis=axis, keepdims=keepdims).shape
        == np.quantile(x, (0.5, 0.8), axis=axis, keepdims=keepdims).shape
    )


@pytest.mark.parametrize("keepdims", [True, False])
def test_return_shapes_without_axis(keepdims):
    rng = np.random.RandomState(0)
    x = rng.randn(100, 10, 1)
    weights = 0.01 * np.ones_like(x)

    assert (
        py_weighted_quantile(x, 0.5, weights, axis=None, keepdims=keepdims).shape
        == np.quantile(x, 0.5, axis=None, keepdims=keepdims).shape
    )

    if not keepdims:
        assert isinstance(
            py_weighted_quantile(x, 0.5, weights, axis=None, keepdims=keepdims),
            (np.float32, float),
        )


def test_errors():
    rng = np.random.RandomState(0)
    x = rng.randn(100, 10, 20)
    weights = 0.01 * np.ones_like(x)

    # axis should be integer
    with pytest.raises(NotImplementedError):
        py_weighted_quantile(x, 0.5, weights, axis=(1, 2))
