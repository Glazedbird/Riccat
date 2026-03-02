import numpy as np

def l2_error(y_pred, y_true):
    diff = y_pred - y_true
    return float(np.sqrt(np.mean(diff * diff)))

def linf_error(y_pred, y_true):
    return float(np.max(np.abs(y_pred - y_true)))