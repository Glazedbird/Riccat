import numpy as np
from scipy.integrate import solve_ivp
from .dre import f_scipy

def solve_reference(t0, t1, x0, t_eval, method="Radau", rtol=1e-10, atol=1e-12):
    sol = solve_ivp(
        fun=f_scipy,
        t_span=(t0, t1),
        y0=np.array([x0], dtype=float),
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
        vectorized=False,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    x = sol.y[0]  # shape [len(t_eval)]
    return x