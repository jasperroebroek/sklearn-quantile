from packaging import version

import sklearn
scikit_version = version.parse(sklearn.__version__)


def create_keyword_dict(estimator, **kwargs):
    if kwargs is None:
        kwargs = {}

    if scikit_version >= version.Version('1.2.0'):
        kwargs['estimator'] = estimator
    else:
        kwargs['base_estimator'] = estimator

    return kwargs
