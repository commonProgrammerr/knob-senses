"""A lib Python do modelo do knob-senses.

Mapa dos modulos (importe o que precisar; este __init__ so' re-exporta o contrato,
para `import model` nao puxar pandas nem sklearn):

    contract     features/ordem, alvo, faixa fisica, limites e cortes do split
    paths        os caminhos do repo, resolvidos do __file__
    data         leitura dos CSVs, PCA_lag1, split, normalizacao
    metrics      evaluate() — MAE/RMSE/R2/vies, com clip
    rf           hiperparametros, treino, varredura e geracao da lib C
    rf_verify    conferidor nativo (g++/xtensa) contra o sklearn (RF e k-NN)
    knn          quantizacao int16, emulador exato, varredura e geracao da lib C do k-NN
    server       o protocolo do servidor de teste (model_teste.ipynb)
"""

from .contract import (
    INDEX,
    LAG,
    LIMITS,
    PCA_MAX,
    PCA_MIN,
    RF_FEATURES,
    SEED,
    SPLIT_COLUMNS,
    SPLIT_CUTS,
    SPLIT_FEATURES,
    TARGET,
    TRAIN_STATION,
)

__all__ = [
    "INDEX",
    "LAG",
    "LIMITS",
    "PCA_MAX",
    "PCA_MIN",
    "RF_FEATURES",
    "SEED",
    "SPLIT_COLUMNS",
    "SPLIT_CUTS",
    "SPLIT_FEATURES",
    "TARGET",
    "TRAIN_STATION",
]
