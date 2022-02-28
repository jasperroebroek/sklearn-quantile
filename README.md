[![Documentation Status](https://readthedocs.org/projects/sklearn-quantile/badge/?version=latest)](https://sklearn-quantile.readthedocs.io/en/latest/?badge=latest)

This module provides quantile machine learning models for python, in a plug-and-play fashion in the sklearn environment. This means that practically the only dependency is sklearn and all its functionality is applicable to the here provided models without code changes.

The models implemented here share the trait that they are trained in exactly the same way as their non-quantile counterpart. The quantile information is only used in the prediction phase. The advantage of this (over for example Gradient Boosting Quantile Regression) is that several quantiles can be predicted at once without the need for retraining the model, which overall leads to a significantly faster workflow. Note that accuracy of doing this depends on the data. As can be seen in the example in the documentation: with certain data characteristics different quantiles might require different parameter optimisation for optimal performance. This is obviously possible with the implemented models here, but this requires the use of a single quantile during prediction, thus losing the speed advantage described above.

For guidance see docs (through the link in the badge). They include an example that for quantile regression forests in exactly the same template as used for Gradient Boosting Quantile Regression in sklearn for comparability.

Implemented:
- Random Forest Quantile Regression 
  - RandomForestQuantileRegressor: the main implementation
  - SampleRandomForestQuantileRegressor: an approximation, that is much faster than the main implementation.
  - RandomForestMaximumRegressor: mathematically equivalent to the main implementation but much faster.

- Extra Trees Quantile Regression
  - ExtraTreesQuantileRegressor: the main implementation
  - SampleExtraTreesQuantileRegressor: an approximation, that is much faster than the main implementation.

- Quantile K-nearest neighbors (KNeighborsQuantileRegressor)

# Installation

The package can be installed with conda:

```
conda install --channel conda-forge sklearn-quantile
```

# Example

An example of Random Forest Quantile Regression in action (both the main implementation and its approximation):

<img src="https://github.com/jasperroebroek/sklearn-quantile/raw/master/docs/source/notebooks/example.png"/>

# Usage example

Random Forest Quantile Regressor predicting the 5th, 50th and 95th percentile of the California housing dataset.

```
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn_quantile import RandomForestQuantileRegressor

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=0)

qrf = RandomForestQuantileRegressor(q=[0.05, 0.50, 0.95])
qrf.fit(X_train, y_train)

y_pred_5, y_pred_median, y_pred_95 = qrf.predict(X_test)
qrf.score(X_test, y_test)
```

# Important links

- API reference: https://sklearn-quantile.readthedocs.io/en/latest/api.html
- Documentation: https://sklearn-quantile.readthedocs.io/en/latest/index.html
