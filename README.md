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

The package can be installed with pip (when cython and scikit-learn are already installed):

```
pip install sklearn-quantile
```

# Example

An example of Random Forest Quantile Regression in action (both the main implementation and its approximation):

<img src="https://github.com/jasperroebroek/sklearn-quantile/raw/master/tests/examples/readme_example.png"/>
