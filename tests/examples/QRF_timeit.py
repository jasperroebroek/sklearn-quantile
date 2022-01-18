import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from sklearn_quantile import RandomForestQuantileRegressor as QRF

from time import time

boston = load_boston()
X, y = boston.data, boston.target

# Test working
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=0)

qrf_default = QRF(n_estimators=100, random_state=0, n_jobs=-1, method='default', quantiles=np.linspace(0, 1, 100))
qrf_default_2o = QRF(n_estimators=100, random_state=0, n_jobs=-1, method='default', quantiles=np.linspace(0, 1, 100))
qrf_sample = QRF(n_estimators=100, random_state=0, n_jobs=-1, method='sample', quantiles=np.linspace(0, 1, 100))

mul = 1

for mul in [1, 10, 100, 1000]:
    print("--------------------------")
    print(455 * mul)

    c_X_train = np.repeat(X_train, mul, axis=0)
    c_y_train = np.repeat(y_train, mul, axis=0)

    if mul < 1000:
        print("default")
        start = time()
        qrf_default.fit(c_X_train, c_y_train)
        print(f"- fit : \t{time() - start}")
        start = time()
        qrf_default.predict(c_X_train)
        print(f"- predict : \t{time() - start}")

    if mul < 1000:
        print("default - 2 outputs")
        start = time()
        qrf_default_2o.fit(c_X_train, np.stack([c_y_train, c_y_train], axis=1))
        print(f"- fit : \t{time() - start}")
        start = time()
        qrf_default.predict(c_X_train)
        print(f"- predict : \t{time() - start}")

    print("sample")
    start = time()
    qrf_sample.fit(c_X_train, c_y_train)
    print(f"- fit : \t{time() - start}")
    start = time()
    qrf_sample.predict(c_X_train)
    print(f"- predict : \t{time() - start}")
