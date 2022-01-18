import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn_quantile import RandomForestQuantileRegressor as QRF
from sklearn_quantile import RandomForestMaximumRegressor as MAX_RF

boston = load_boston()
X, y = boston.data, boston.target

# Test working
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=0)

for ii in [1, 20]:
    qrf_default = QRF(n_estimators=100, random_state=0, n_jobs=-1, quantiles=1, method='default', min_samples_leaf=ii)
    qrf_sample = QRF(n_estimators=100, random_state=0, n_jobs=-1, quantiles=1, method='sample', min_samples_leaf=ii)
    max_rf = MAX_RF(n_estimators=100, random_state=0, n_jobs=-1, min_samples_leaf=ii)

    qrf_sample.fit(X_train, y_train)
    qrf_default.fit(X_train, y_train)
    max_rf.fit(X_train, y_train)

    f, ax = plt.subplots(ncols=2, nrows=2)
    for i, q in enumerate([0.95, 0.99, 0.999, 1]):
        ax = ax.flatten()
        qrf_default.quantiles = q
        ax[i].scatter(qrf_default.predict(X_train), max_rf.predict(X_train))
        ax[i].set_title(q)
    f.suptitle(f"Default, samples per leaf = {ii}")
    plt.show()

    f, ax = plt.subplots(ncols=2, nrows=2)
    for i, q in enumerate([0.95, 0.99, 0.999, 1]):
        ax = ax.flatten()
        qrf_sample.quantiles = q
        ax[i].scatter(qrf_sample.predict(X_train), max_rf.predict(X_train))
        ax[i].set_title(q)
    f.suptitle(f"Sample, samples per leaf = {ii}")
    plt.show()

    print(np.allclose(qrf_default.predict(X_train), max_rf.predict(X_train)))
    print(np.allclose(qrf_sample.predict(X_train), max_rf.predict(X_train)))
