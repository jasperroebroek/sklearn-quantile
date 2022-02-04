import numpy as np
import pytest

from sklearn_quantile.utils import weighted_quantile

from numpy.testing import assert_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal
from numpy.testing import assert_raises


def test_quantile_equal_weights():
    rng = np.random.RandomState(0)
    x = rng.randn(10)
    weights = 0.1 * np.ones(10)

    # since weights are equal, quantiles lie in the midpoint.
    sorted_x = np.sort(x)
    expected = 0.5 * (sorted_x[1:] + sorted_x[:-1])
    actual = np.asarray([weighted_quantile(x, q, weights) for q in np.arange(0.1, 1.0, 0.1)])

    assert_array_almost_equal(expected, actual)

    # check quantiles at (0.05, 0.95) at intervals of 0.1
    actual = np.asarray([weighted_quantile(x, q, weights) for q in np.arange(0.05, 1.05, 0.1)])
    assert_array_almost_equal(sorted_x, actual)

    # it should be the same the calculated all quantiles at the same time instead of looping over them
    assert_array_almost_equal(actual, weighted_quantile(x, weights=weights, q=np.arange(0.05, 1.05, 0.1)))


def test_quantile_toy_data():
    x = [1, 2, 3]
    weights = [1, 4, 5]

    assert_equal(weighted_quantile(x, 0.0, weights), 1)
    assert_equal(weighted_quantile(x, 1.0, weights), 3)

    assert_equal(weighted_quantile(x, 0.05, weights), 1)
    assert_almost_equal(weighted_quantile(x, 0.30, weights), 2)
    assert_equal(weighted_quantile(x, 0.75, weights), 3)
    assert_almost_equal(weighted_quantile(x, 0.50, weights), 2.44, 2)


@pytest.mark.parametrize('q', [0, 0.1, 0.5, 0.9, 1])
def test_zero_weights(q):
    x = [1, 2, 3, 4, 5]
    w = [0, 0, 0, 0.1, 0.1]

    assert_equal(
        weighted_quantile(x, q, w),
        weighted_quantile([4, 5], q, [0.1, 0.1])
    )


@pytest.mark.parametrize("keepdims", [True, False])
def test_return_shapes(keepdims):
    rng = np.random.RandomState(0)
    x = rng.randn(100, 10, 20)
    weights = 0.01 * np.ones_like(x)

    # shape should be the same as the output of np.quantile. Without weights it is actually the same calculation
    assert (
        weighted_quantile(x, 0.5, weights, axis=0, keepdims=keepdims).shape ==
        np.quantile(x, 0.5, axis=0, keepdims=keepdims).shape
    )
    assert (
        weighted_quantile(x, 0.5, weights, axis=1, keepdims=keepdims).shape ==
        np.quantile(x, 0.5, axis=1, keepdims=keepdims).shape
    )
    assert (
        weighted_quantile(x, 0.5, weights, axis=2, keepdims=keepdims).shape ==
        np.quantile(x, 0.5, axis=2, keepdims=keepdims).shape
    )
    assert (
        weighted_quantile(x, (0.5, 0.8), weights, axis=0, keepdims=keepdims).shape ==
        np.quantile(x, (0.5, 0.8), axis=0, keepdims=keepdims).shape
    )
    if keepdims:
        assert (
            weighted_quantile(x, 0.5, weights, axis=None, keepdims=keepdims).shape ==
            np.quantile(x, 0.5, axis=None, keepdims=keepdims).shape
        )
    else:
        assert isinstance(weighted_quantile(x, 0.5, weights, axis=None, keepdims=keepdims), (np.float32, float))


@pytest.mark.parametrize("keepdims", [True, False])
def test_return_shapes_empty_dims(keepdims):
    rng = np.random.RandomState(0)
    x = rng.randn(1, 100, 1)
    weights = 0.01 * np.ones_like(x)

    assert (
        weighted_quantile(x, 0.5, weights, axis=1, keepdims=keepdims).shape ==
        np.quantile(x, 0.5, axis=1, keepdims=keepdims).shape
    )
    assert (
        weighted_quantile(x, 0.5, weights=None, axis=1, keepdims=keepdims).shape ==
        np.quantile(x, 0.5, axis=1, keepdims=keepdims).shape
    )

    if keepdims:
        assert (
            weighted_quantile(x, 0.5, weights, keepdims=keepdims).shape ==
            np.quantile(x, 0.5, keepdims=keepdims).shape
        )


def test_errors():
    rng = np.random.RandomState(0)
    x = rng.randn(100, 10, 20)
    weights = 0.01 * np.ones_like(x)

    # axis should be integer
    assert_raises(NotImplementedError, weighted_quantile, x, 0.5, weights, axis=(1, 2))
