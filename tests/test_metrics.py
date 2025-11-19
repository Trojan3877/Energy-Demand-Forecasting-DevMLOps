import numpy as np
from src.utils_metrics import rmse, mae, mape, r2

def test_rmse():
    assert rmse(np.array([1,2,3]), np.array([1,2,3])) == 0

def test_mae():
    assert mae(np.array([1,2,3]), np.array([1,2,3])) == 0

def test_mape():
    assert mape(np.array([100]), np.array([110])) > 0

def test_r2():
    assert r2(np.array([1,2,3]), np.array([1,2,3])) == 1
