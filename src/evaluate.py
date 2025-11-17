
"""evaluate.py
Common evaluation helpers (skeleton).
"""
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)
