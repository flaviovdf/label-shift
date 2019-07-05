# -*- coding: utf8


from sklearn.metrics import confusion_matrix


import numpy as np


def calculate_marginal(y, n_classes):
    mu = np.zeros(shape=(n_classes, 1))
    for i in range(n_classes):
        mu[i] = np.sum(y == i)
    return mu / y.shape[0]


def estimate_labelshift_ratio(y_true_val, y_pred_val, y_pred_trn, n_classes):
    labels = np.arange(n_classes)
    C = confusion_matrix(y_true_val, y_pred_val, labels).T
    C = C / y_true_val.shape[0]

    mu_t = calculate_marginal(y_pred_trn, n_classes)
    lamb = 1.0 / min(y_pred_val.shape[0], y_pred_trn.shape[0])

    I = np.eye(n_classes)
    wt = np.linalg.solve(np.dot(C.T, C) + lamb * I, np.dot(C.T, mu_t))
    return wt


def estimate_target_dist(wt, y_true_val, n_classes):
    mu_t = calculate_marginal(y_true_val, n_classes)
    return wt * mu_t
