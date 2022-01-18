import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn_quantile import RandomForestQuantileRegressor
from sklearn_quantile import ExtraTreesQuantileRegressor

boston = load_boston()
X, y = boston.data, boston.target
# y = np.vstack([y, y]).T # does not work for gradient boosting

# Test working
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=0)

qrf_default = RandomForestQuantileRegressor(random_state=0)
qrf_rs = RandomForestQuantileRegressor(random_state=0, method='sample', n_jobs=-1)
rf = RandomForestRegressor(random_state=0, criterion='mae')
gbr = GradientBoostingRegressor(random_state=0, loss='quantile', alpha=0.9)

qrf_default.fit(X_train, y_train)
qrf_rs.fit(X_train, y_train)
rf.fit(X_train, y_train)
gbr.fit(X_train, y_train)

f, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
ax = ax.flatten()

qrf_default.quantiles = 0.5
qrf_rs.quantiles = 0.5

ax[0].scatter(qrf_default.predict(X_test), rf.predict(X_test))
ax[0].set_xlabel("Quantile regression forest - median")
ax[0].set_ylabel("Random forest with MAE criterion")
ax[0].set_title("Default implementation (from original paper)")
ax[0].plot([10, 50], [10, 50])

ax[1].scatter(qrf_rs.predict(X_test), rf.predict(X_test))
ax[1].set_xlabel("Quantile regression forest - median")
ax[1].set_ylabel("Random forest with MAE criterion")
ax[1].set_title("Random sample implementation (as in quantregForest)")
ax[1].plot([10, 50], [10, 50])

qrf_default.quantiles = 0.9
qrf_rs.quantiles = 0.9

ax[2].scatter(qrf_default.predict(X_test), gbr.predict(X_test))
ax[2].set_xlabel("Quantile regression forest - q=0.9")
ax[2].set_ylabel("GradientBoostRegressor - q=0.9")
ax[2].set_title("Default implementation (from original paper)")
ax[2].plot([10, 50], [10, 50])

ax[3].scatter(qrf_rs.predict(X_test), gbr.predict(X_test))
ax[3].set_xlabel("Quantile regression forest - q=0.9")
ax[3].set_ylabel("GradientBoostRegressor - q=0.9")
ax[3].set_title("Random sample implementation (as in quantregForest)")
ax[3].plot([10, 50], [10, 50])

plt.show()
