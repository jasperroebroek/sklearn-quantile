.. currentmodule:: sklearn_quantile

Quantile Regression Forest
==========================

Quantile regression forests (and similarly Extra Trees Quantile Regression Forests) are based on the paper by
Meinshausen (2006). The training of the model is based on a MSE criterion, which is the same as for standard regression
forests, but prediction calculates weighted quantiles on the ensemble of all predicted leafs. This allows for prediction
to be done on several quantiles at once, without the need for retraining the model.


Quantile KNN
============

Quantile KNN is similar to the Quantile Regression Forests, as the training of the model is non quantile
dependent, thus predictions can be made for several quantiles at the time. This speeds up the workflow significantly.
The model implemented here is strictly based on the standard KNN, thus all parameterisations and options are identical.
