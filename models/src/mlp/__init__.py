"""O modelo MLP (Keras) do knob-senses: rede, treino e export para o Edge Impulse."""

from .core import (
    BATCH,
    ENTRADAS,
    EPOCHS,
    OCULTAS,
    PACIENCIA,
    build,
    export,
    prepara,
    train,
)

__all__ = [
    "BATCH",
    "ENTRADAS",
    "EPOCHS",
    "OCULTAS",
    "PACIENCIA",
    "build",
    "export",
    "prepara",
    "train",
]
