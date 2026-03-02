import os
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from .config import DREConfig
from .reference import solve_reference
from .nn_models import MLP
from .metrics import l2_error, linf_error
from .dre import f_torch

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

def train_supervised_plus_residual(cfg: DREConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    # 训练点（同时用于监督与残差）
    t_train = np.linspace(cfg.t0, cfg.t1, cfg.n_train)
    x_train = solve_reference(cfg.t0, cfg.t1, cfg.x0, t_train,
                              method=cfg.ref_method, rtol=cfg.rtol, atol=cfg.atol)

    # 评估点
    t_eval = np.linspace(cfg.t0, cfg.t1, cfg.n_ref_eval)
    x_ref = solve_reference(cfg.t0, cfg.t1, cfg.x0, t_eval,
                            method=cfg.ref_method, rtol=cfg.rtol, atol=cfg.atol)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(hidden=cfg.hidden, depth=cfg.depth, activation=cfg.activation).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    mse = torch.nn.MSELoss()

    t_train_t = torch.tensor(t_train, dtype=torch.float32, device=device).view(-1, 1)
    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device).view(-1, 1)

    start = time.perf_counter()
    hist = []

    for ep in range(1, cfg.epochs + 1):
        model.train()

        # 监督项
        x_pred = model(t_train_t)
        loss_sup = mse(x_pred, x_train_t)

        # 残差项：需要 t 可求导
        t_req = t_train_t.clone().detach().requires_grad_(True)
        x_req = model(t_req)
        dx_dt = torch.autograd.grad(
            outputs=x_req,
            inputs=t_req,
            grad_outputs=torch.ones_like(x_req),
            create_graph=True
        )[0]
        res = dx_dt - f_torch(t_req, x_req)
        loss_res = torch.mean(res * res)

        loss = cfg.sup_weight * loss_sup + cfg.res_weight * loss_res

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % 200 == 0 or ep == 1:
            hist.append((ep, float(loss_sup.item()), float(loss_res.item()), float(loss.item())))

    train_time = time.perf_counter() - start

    # 推断 + 指标
    model.eval()
    with torch.no_grad():
        t_eval_t = torch.tensor(t_eval, dtype=torch.float32, device=device).view(-1, 1)
        x_pred_eval = model(t_eval_t).cpu().numpy().reshape(-1)

    e2 = l2_error(x_pred_eval, x_ref)
    einf = linf_error(x_pred_eval, x_ref)

    # 保存曲线
    df = pd.DataFrame({
        "t": t_eval,
        "x_ref": x_ref,
        "x_pred": x_pred_eval,
        "abs_err": np.abs(x_pred_eval - x_ref)
    })
    csv_path = os.path.join(cfg.out_dir, "sup_plus_res_curve.csv")
    df.to_csv(csv_path, index=False)

    # 图：解
    plt.figure()
    plt.plot(t_eval, x_ref, label="reference")
    plt.plot(t_eval, x_pred_eval, label="NN (sup+res)")
    plt.legend()
    plt.xlabel("t"); plt.ylabel("x(t)")
    plt.title("Solution curve")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "sup_plus_res_solution.png"), dpi=200)
    plt.close()

    # 图：误差
    plt.figure()
    plt.plot(t_eval, np.abs(x_pred_eval - x_ref))
    plt.xlabel("t"); plt.ylabel("|error|")
    plt.title("Absolute error")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "sup_plus_res_error.png"), dpi=200)
    plt.close()

    # 图：训练损失
    hist_df = pd.DataFrame(hist, columns=["epoch", "loss_sup", "loss_res", "loss_total"])
    hist_df.to_csv(os.path.join(cfg.out_dir, "sup_plus_res_trainlog.csv"), index=False)

    summary = {
        "mode": "sup+residual",
        "ref_method": cfg.ref_method,
        "epochs": cfg.epochs,
        "train_time_sec": train_time,
        "l2_error": e2,
        "linf_error": einf,
        "csv": csv_path,
    }
    return summary

if __name__ == "__main__":
    cfg = DREConfig()
    out = train_supervised_plus_residual(cfg)
    print(out)