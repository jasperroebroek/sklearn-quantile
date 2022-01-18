import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor
from sklearn_quantile import RandomForestQuantileRegressor
from sklearn_quantile import KNeighborsQuantileRegressor

boston = load_boston()
X, y = boston.data, boston.target
# y = np.vstack([y, y]).T # does not work for gradient boosting

# Test working
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=0)

knn = KNeighborsRegressor()
qknn = KNeighborsQuantileRegressor()
qrf = RandomForestQuantileRegressor(random_state=0)

knn.fit(X_train, y_train)
qknn.fit(X_train, y_train)
qrf.fit(X_train, y_train)

f, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
ax = ax.flatten()

qknn.quantiles = 0.5
qrf.quantiles = 0.5

ax[0].scatter(qknn.predict(X_test), knn.predict(X_test))
ax[0].set_xlabel("Quantile KNN - median")
ax[0].set_ylabel("KNN - mean")
ax[0].plot([10, 50], [10, 50])

ax[1].scatter(qknn.predict(X_test), qrf.predict(X_test))
ax[1].set_xlabel("Quantile KNN - median")
ax[1].set_ylabel("Quantile regression forest - median")
ax[1].plot([10, 50], [10, 50])

qknn.quantiles = 0.9
qrf.quantiles = 0.9

ax[2].scatter(qknn.predict(X_test), knn.predict(X_test))
ax[2].set_xlabel("Quantile KNN - q=0.9")
ax[2].set_ylabel("KNN - mean")
ax[2].plot([10, 50], [10, 50])

ax[3].scatter(qknn.predict(X_test), qrf.predict(X_test))
ax[3].set_xlabel("Quantile KNN - q=0.9")
ax[3].set_ylabel("Quantile regression forest - q=0.9")
ax[3].plot([10, 50], [10, 50])

plt.show()
