"""As metricas do projeto — em kW (R2 e' adimensional)."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import data


def evaluate(name: str, y, pred, pca_min: float | None = None,
             pca_max: float | None = None) -> dict:
    """MAE/RMSE em kW e R2 adimensional. Reporta tambem com clip na faixa fisica.

    Sem `pca_min`/`pca_max`, a faixa vem do dataset (`data.target_limits`). Unica
    funcao da lib que imprime tabela (as outras devolvem dados): manter o print aqui
    preserva a cara da saida dos notebooks.
    """
    if pca_min is None or pca_max is None:
        pca_min, pca_max = data.target_limits()
    pred_clip = np.clip(pred, pca_min, pca_max)
    row = {
        "modelo": name,
        "MAE (kW)": mean_absolute_error(y, pred),
        "RMSE (kW)": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": r2_score(y, pred),
        "MAE com clip (kW)": mean_absolute_error(y, pred_clip),
        "vies (kW)": float((pred - y).mean()),
    }
    print(f"{name:20} MAE {row['MAE (kW)']:.4f} kW | RMSE {row['RMSE (kW)']:.4f} kW "
          f"| R2 {row['R2']:.4f} | vies {row['vies (kW)']:+.4f} kW")
    return row
