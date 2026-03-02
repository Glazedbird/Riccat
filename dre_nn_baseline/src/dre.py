import numpy as np
import torch

def a_t(t):
    # 可改：例如常数或随时间变化
    return 1.0

def b_t(t):
    return -1.0

def c_t(t):
    return 0.2

# ---- SciPy 用：输入 t(float), x(np.ndarray shape (1,) or scalar) 输出 dx/dt ----
def f_scipy(t, x):
    # 兼容 solve_ivp 传进来的 x（可能是 shape (1,)）
    x_val = float(np.atleast_1d(x)[0])
    return a_t(t) * x_val * x_val + b_t(t) * x_val + c_t(t)

# ---- PyTorch 用：输入 t(tensor shape [N,1]), x(tensor shape [N,1]) 输出 dx/dt ----
def f_torch(t, x):
    # t, x 都是 tensor
    a = torch.ones_like(t) * a_t(0.0)  # 若 a_t 依赖 t，可在这里写成 a_t_tensor(t)
    b = torch.ones_like(t) * b_t(0.0)
    c = torch.ones_like(t) * c_t(0.0)
    return a * x * x + b * x + c