import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn_quantile import (
    RandomForestQuantileRegressor,
    SampleRandomForestQuantileRegressor,
)


def f(x):
    """The function to predict."""
    return x * np.sin(x)


rng = np.random.RandomState(42)
X = np.atleast_2d(rng.uniform(0, 10.0, size=1000)).T
expected_y = f(X).ravel()

sigma = 0.5 + X.ravel() / 10
noise = rng.lognormal(sigma=sigma) - np.exp(sigma ** 2 / 2)
y = expected_y + noise

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

common_params = dict(
    max_depth=3,
    min_samples_leaf=4,
    min_samples_split=4,
)
qrf = RandomForestQuantileRegressor(**common_params, q=[0.05, 0.5, 0.95])
qrf.fit(X_train, y_train)

sqrf = SampleRandomForestQuantileRegressor(**common_params, q=[0.05, 0.5, 0.95])
sqrf.fit(X_train, y_train)

rf = RandomForestRegressor(**common_params)
rf.fit(X_train, y_train)

xx = np.atleast_2d(np.linspace(0, 10, 1000)).T

predictions = qrf.predict(xx)
s_predictions = sqrf.predict(xx)

y_pred = rf.predict(xx)

y_lower = predictions[0]
y_med = predictions[1]
y_upper = predictions[2]

y_s_lower = s_predictions[0]
y_s_med = s_predictions[1]
y_s_upper = s_predictions[2]

fig = plt.figure(figsize=(10, 10))
plt.plot(xx, f(xx), 'g', linewidth=3, label=r'$f(x) = x\,\sin(x)$')
plt.plot(X_test, y_test, 'b.', markersize=10, label='Test observations')

plt.plot(xx, y_med, 'r-', label='Predicted median', color="orange")
plt.plot(xx, y_s_med, 'r-', label='Aproximation predicted median', color="orange", linestyle="--")
plt.plot(xx, y_pred, 'r-', label='Predicted mean')

plt.plot(xx, y_upper, 'g', label='Predicted 95th percentile')
plt.plot(xx, y_s_upper, 'g--', alpha=0.8, label='Approximated 95th percentile')
plt.plot(xx, y_lower, 'grey', label='Predicted 5th percentile')
plt.plot(xx, y_s_lower, 'grey', linestyle='--', alpha=0.8, label='Approximated 5th percentile')

plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 25)
plt.legend(loc='upper left')
plt.savefig("tests/examples/readme_example.png", dpi=300, bbox_inches='tight')
plt.show()