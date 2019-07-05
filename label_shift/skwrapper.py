# -*- coding: utf8

from . import calculate_marginal
from . import estimate_labelshift_ratio
from . import estimate_target_dist

from scipy import stats as ss


import numpy as np


class LabelShiftDetectorSKLearn(object):


    def __init__(self, estimator, validation_proportion=0.5, shuffle=True,
                 sig=0.05):
        self.estimator = estimator
        self.validation_proportion = validation_proportion
        self.shuffle = shuffle
        self.sig = sig


    def fit(self, X, y):
        X = np.asanyarray(X, dtype='d')
        y = np.asanyarray(y, dtype='i')

        n = len(X)
        n_classes = len(set(y))
        self.n_classes_ = n_classes

        idx = np.arange(n)
        if self.shuffle:
            np.random.shuffle(idx)

        k = int(n * (1-self.validation_proportion))
        X_trn = X[idx[k:]]
        y_trn = y[idx[k:]]

        X_val = X[idx[:k]]
        y_val = y[idx[:k]]

        _, p = ss.ks_2samp(y_trn, y_val)
        if p < self.sig:
            raise Exception('A label shift exists in the training set.')

        self.estimator = self.estimator.fit(X_trn, y_trn)
        y_pred_trn = self.estimator.predict(X_trn)
        y_pred_val = self.estimator.predict(X_val)
        self.y_pred_val_ = y_pred_val

        self.wt_est_ = estimate_labelshift_ratio(y_val, y_pred_val, y_pred_trn,
                                                 n_classes)
        self.py_est_ = estimate_target_dist(self.wt_est_, y_pred_val,
                                            n_classes)
        self.py_base_ = calculate_marginal(y_val, n_classes)

        return self


    def predict(self, X):
        if self.estimator is None:
            raise Exception('Fit was not yet called')

        return self.estimator.predict(X)


    def label_shift_detector(self, X, y=None, return_bootstrap=False,
                             bootstrap_size=500):
        if self.estimator is None:
            raise Exception('Fit was not yet called')

        y_pred = self.predict(X)
        _, no_boot = ss.ks_2samp(self.y_pred_val_, y_pred)
        if return_bootstrap:
            results = []
            for _ in range(bootstrap_size):
                y_boot_v = np.random.choice(self.y_pred_val_,
                                            size=len(self.y_pred_val_),
                                            replace=True)
                y_boot_p = np.random.choice(y_pred,
                                            size=len(y_pred),
                                            replace=True)

                _, p = ss.ks_2samp(y_boot_p, y_boot_v)
                results.append(p)
            results = np.array(results)

        if y is not None:
            py_true = calculate_marginal(y, self.n_classes_)
            wt_true = py_true / self.py_base_
            norm = np.power(self.wt_est_ - wt_true, 2).sum()
            kld = ss.entropy(py_true, self.py_est_)[0]
            if return_bootstrap:
                return no_boot, results, norm, kld
            else:
                return no_boot, norm, kld
        else:
            if return_bootstrap:
                return no_boot, results
            else:
                return no_boot
