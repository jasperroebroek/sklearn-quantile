import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn_quantile import RandomForestQuantileRegressor as QRF
from skgarden_base.quantile.ensemble import RandomForestQuantileRegressor as QRF_base

boston = load_boston()
X, y = boston.data, boston.target

# Test working
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=0)

for ii in [1, 20]:
    qrf_base = QRF_base(n_estimators=100, random_state=0, n_jobs=-1, min_samples_leaf=ii)
    qrf_default = QRF(n_estimators=100, method='default', random_state=0, n_jobs=-1, min_samples_leaf=ii)
    qrf_sample = QRF(n_estimators=100, method='sample', random_state=0, n_jobs=-1, min_samples_leaf=ii)

    qrf_base.fit(X_train, y_train)
    qrf_default.fit(X_train, y_train)
    qrf_sample.fit(X_train, y_train)

    f, ax = plt.subplots(ncols=2, nrows=2)
    for i, q in enumerate([0.5, 0.95, 0.99, 0.999]):
        ax = ax.flatten()
        qrf_default.quantiles = q
        ax[i].scatter(qrf_default.predict(X_train), qrf_base.predict(X_train, quantile=q*100))
        ax[i].set_title(q)
    f.suptitle(f"Default, samples per leaf = {ii}")
    plt.show()

    qrf_default.fit(X_train, np.stack([y_train, y_train], axis=1))
    f, ax = plt.subplots(ncols=2, nrows=2)
    for i, q in enumerate([0.5, 0.95, 0.99, 0.999]):
        ax = ax.flatten()
        qrf_default.quantiles = q
        ax[i].scatter(qrf_default.predict(X_train)[:, 0], qrf_base.predict(X_train, quantile=q*100))
        ax[i].set_title(q)
    f.suptitle(f"Default 2 outputs, samples per leaf = {ii}")
    plt.show()

    f, ax = plt.subplots(ncols=2, nrows=2)
    for i, q in enumerate([0.5, 0.95, 0.99, 0.999]):
        ax = ax.flatten()
        qrf_sample.quantiles = q
        ax[i].scatter(qrf_sample.predict(X_train), qrf_base.predict(X_train, quantile=q*100))
        ax[i].set_title(q)
    f.suptitle("Sample")
    plt.show()
