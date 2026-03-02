import os
import json
import pandas as pd

from .config import DREConfig
from .train_supervised import train_supervised
from .train_residual import train_supervised_plus_residual

def main():
    cfg = DREConfig()
    os.makedirs(cfg.out_dir, exist_ok=True)

    results = []
    results.append(train_supervised(cfg))
    results.append(train_supervised_plus_residual(cfg))

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(cfg.out_dir, "summary.csv"), index=False)

    with open(os.path.join(cfg.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(df)

if __name__ == "__main__":
    main()