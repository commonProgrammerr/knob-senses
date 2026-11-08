"""Gaps artificiais no split de teste: a validacao do soft-sensor como solucao.

Cenario: a telemetria do inversor (PCA) falha por intervalos, mas as variaveis
meteorologicas (ghit, temperatura, etc.) vem de outro sistema e continuam
disponiveis. O gap oculta SO o PCA; o soft-sensor estima a potencia durante o
buraco, e o valor real escondido serve de ground truth para medir o erro DENTRO
dos gaps — o numero que decide se o soft-sensor e' viavel como alternativa a
medidores dedicados.
"""

import numpy as np
import pandas as pd

from . import contract

# Quantas horas vale um bin de cada granularidade do split (coluna `freq`).
HORAS_POR_BIN = {"5min": 5 / 60, "10min": 10 / 60, "1h": 1.0, "1D": 24.0}

# Duracao dos blocos contiguos, em horas, e quantos blocos de cada duracao.
# Um bloco so' e' criado se a duracao couber na granularidade (>= 1 bin).
GAP_HORAS = (1, 6, 24, 72)
GAP_QUANTIDADE = 4

# Fracao das linhas NAO-gap removidas a esmo: falhas esporadicas, sem bloco.
FRACAO_ALEATORIA = 0.10


def gerar_gaps(df: pd.DataFrame, freq: str,
               horas=GAP_HORAS, quantidade=GAP_QUANTIDADE,
               fracao_aleatoria=FRACAO_ALEATORIA,
               seed=contract.SEED) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Devolve (df_gapado, mascara_gap, rotulo_gap) para UMA usina e UMA granularidade.

    Os blocos contiguos sao posicionados por sorteio (sem sobrepor outros
    blocos, com uma linha de respiro) e a fracao aleatoria cai nas linhas que
    sobraram. O PCA das linhas em gap vira NaN. `mascara_gap` marca as linhas
    ocultas (True = gap) e `rotulo_gap` guarda a duracao do bloco em horas, ou
    -1 para as falhas esporadicas (fracao aleatoria), ou 0 fora dos gaps — e'
    o que permite desagregar as metricas por duracao.
    """
    df = df.reset_index(drop=True).copy()
    n = len(df)
    rng = np.random.default_rng(seed)
    mascara = np.zeros(n, dtype=bool)
    rotulo = np.zeros(n, dtype=int)

    horas_bin = HORAS_POR_BIN[freq]
    for duracao_h in horas:
        comprimento = int(np.ceil(duracao_h / horas_bin))
        if comprimento < 1 or comprimento >= n:
            continue                     # duracao menor que o bin: nao faz sentido aqui
        for _ in range(quantidade):
            tentativas = 0
            while tentativas < 50:
                inicio = int(rng.integers(0, n - comprimento))
                lo = max(0, inicio - 1)
                hi = min(n, inicio + comprimento + 1)
                if not mascara[lo:hi].any():
                    break       # bloco livre, com 1 linha de respiro dos demais
                tentativas += 1
            mascara[inicio:inicio + comprimento] = True
            rotulo[inicio:inicio + comprimento] = duracao_h

    if fracao_aleatoria:
        restantes = np.flatnonzero(~mascara)
        escolhidos = rng.choice(restantes,
                                size=int(len(restantes) * fracao_aleatoria),
                                replace=False)
        mascara[escolhidos] = True
        rotulo[escolhidos] = -1          # falha esporadica, sem bloco

    saida = df.copy()
    saida.loc[mascara, contract.TARGET] = np.nan
    return saida, mascara, rotulo
