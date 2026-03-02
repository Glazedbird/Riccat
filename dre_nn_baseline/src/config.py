from dataclasses import dataclass

@dataclass
class DREConfig:
    # 时间区间与采样
    t0: float = 0.0
    t1: float = 5.0
    n_ref_eval: int = 2001      # 画图/评估用的密网格
    n_train: int = 256          # 监督训练采样点数

    # 初值
    x0: float = 0.1

    # 参考解（ODE solver）设置
    ref_method: str = "Radau"   # "RK45", "DOP853", "Radau", "BDF"
    rtol: float = 1e-10
    atol: float = 1e-12

    # NN 设置
    hidden: int = 64
    depth: int = 4
    activation: str = "tanh"    # "tanh" or "relu"

    # 训练设置
    seed: int = 42
    lr: float = 1e-3
    epochs: int = 5000
    batch_size: int = 256       # 标量问题：直接 full-batch 也行

    # 残差（物理项）权重：监督+残差；纯PINN则把 sup_weight 设为0
    sup_weight: float = 1.0
    res_weight: float = 1.0

    # 输出
    out_dir: str = "outputs"