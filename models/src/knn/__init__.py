"""O modelo k-NN (emlearn): quantizacao, emulador, varredura e geracao da lib C."""

from .core import (
    K,
    N_PROTOTYPES,
    SWEEP_K,
    SWEEP_N,
    Y_SCALE,
    Z_CLAMP,
    Z_SCALE,
    _prototipos,
    escolhe_config,
    generate_c_lib,
    preve_emulado,
    preve_emulado_lote,
    quantiza_alvo,
    quantiza_z,
    quantiza_z_c,
    sweep,
)

__all__ = [
    "K",
    "N_PROTOTYPES",
    "SWEEP_K",
    "SWEEP_N",
    "Y_SCALE",
    "Z_CLAMP",
    "Z_SCALE",
    "_prototipos",
    "escolhe_config",
    "generate_c_lib",
    "preve_emulado",
    "preve_emulado_lote",
    "quantiza_alvo",
    "quantiza_z",
    "quantiza_z_c",
    "sweep",
]
