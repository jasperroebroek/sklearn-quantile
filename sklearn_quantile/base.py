import numpy as np
from numpy.lib.function_base import _quantile_is_valid
from sklearn.metrics import mean_pinball_loss


class QuantileRegressorMixin:
    def validate_quantiles(self):
        """
        Validate the quantiles inserted in the quantile regressor
        """
        if not hasattr(self, 'q'):
            raise NotImplementedError("This method can't be used for non quantile-regressors")
        if self.q is None:
            raise AttributeError("Quantiles are not set. Please provide them at setup or with `model.q = q`, with q"
                                 " being a float between 0 and 1, or a an array with the same bounds")
        q = np.asarray(self.q, dtype=np.float32)
        q = np.atleast_1d(q)
        if q.ndim > 2:
            raise ValueError("q must be a scalar or 1D")

        if not _quantile_is_valid(q):
            raise ValueError("Quantiles must be in the range [0, 1]")

        return q

    def score(self, X, y):
        """
        Mean pinball loss for the quantile regressors.

        The average over all quantiles is calculated when more than one is provided.
        """
        q = self.validate_quantiles()
        y_pred = self.predict(X)
        losses = np.empty(q.size)
        for i in range(q.size):
            losses[i] = mean_pinball_loss(y, y_pred[i], alpha=q[i])
        return np.mean(losses)
