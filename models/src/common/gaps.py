"""Gaps artificiais no split de teste: a validacao do soft-sensor como solucao.

Cenario: a telemetria do inversor (PCA) falha por intervalos, mas as variaveis
meteorologicas (ghit, temperatura, etc.) vem de outro sistema e continuam
disponiveis. O gap oculta SO o PCA; o soft-sensor estima a potencia durante o
buraco, e o valor real escondido serve de ground truth para medir o erro DENTRO
dos gaps, o numero que decide se o soft-sensor e' viavel como alternativa a
medidores dedicados.
"""

import numpy as np
import pandas as pd

from . import contract

# Quantas horas vale um bin de cada granularidade do split (coluna `freq`).
HORAS_POR_BIN = {"5min": 5 / 60, "10min": 10 / 60, "1h": 1.0, "1D": 24.0}

# Intervalo entre leituras consecutivas acima disto e' gap real, nao a pausa
# noturna normal (13-14 h no teste). Blocos artificiais nao cobrem esses gaps.
LIMIAR_GAP_REAL = pd.Timedelta(hours=16)

# Blocos menores que isto devem caber dentro do dia em que comecam; nao podem
# entrar na pausa noturna. Blocos maiores podem atravessar a noite.
LIMIAR_GAP_NOTURNO = pd.Timedelta(hours=12)


def _como_timedelta(valor) -> pd.Timedelta:
    """Converte '3h', '2d', Timedelta ou um numero (em horas) em Timedelta."""
    if isinstance(valor, (int, float, np.integer, np.floating)):
        return pd.Timedelta(hours=float(valor))
    return pd.Timedelta(valor)


def _como_lista(gaps):
    """Um intervalo sozinho vira tupla de um; uma sequencia fica como esta."""
    escalar = isinstance(gaps, (str, pd.Timedelta, np.timedelta64,
                                    int, float, np.integer, np.floating))
    return (gaps,) if escalar else tuple(gaps)


def gerar_gaps(df: pd.DataFrame, freq: str, gaps, time_col=contract.INDEX,
               seed=contract.SEED,
               limiar_gap_real=LIMIAR_GAP_REAL
               ) -> tuple[np.ndarray, np.ndarray]:
    """Devolve (mascara_gap, rotulo_gap) para UMA usina e UMA granularidade.

    Cada item de `gaps` (ex.: "3h", "3d") cria UM bloco, com duracao fixa
    de tempo, posicionado em um local sorteado da serie. Para mais de um bloco
    da mesma duracao, repita o valor na lista. O bloco cobre todas as linhas com
    data no intervalo [inicio, inicio + duracao): em uma serie diurna, 24 h de
    gap escondem as leituras de um dia, nao 24 linhas. Os blocos nao se
    sobrepoem, guardam uma linha de respiro dos demais e nao cobrem gaps reais.
    Blocos com duracao menor que 12 h ficam contidos na janela de leituras do
    dia em que comecam; os maiores podem atravessar a pausa noturna.

    `limiar_gap_real` define gap real como o intervalo entre leituras
    consecutivas maior que esse valor. O default de 16 h preserva a pausa
    noturna (13 a 14 h) e rejeita dias inteiros ausentes. Passe None para
    desligar a checagem.

    `mascara_gap` marca as linhas ocultas (True = gap) e `rotulo_gap` guarda a
    duracao do bloco em horas (float), ou 0 fora dos gaps. O dataframe de
    entrada NAO e' alterado: o PCA continua com os valores originais. Numeros
    em `gaps` sao interpretados como horas.
    """
    df = df.reset_index(drop=True).copy()
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)

    tempos_serie = pd.to_datetime(df[time_col])
    if tempos_serie.isna().any():
        raise ValueError("gaps: ha datas nulas na coluna de indice")
    # Trabalhar em ordem temporal e' o que permite medir o bloco em TEMPO, nao
    # em quantidade de linhas; a mascara volta para a ordem original no fim.
    ordem = np.argsort(tempos_serie.to_numpy(), kind="stable")
    tempos = pd.DatetimeIndex(tempos_serie.to_numpy()[ordem])

    rng = np.random.default_rng(seed)
    mascara_ord = np.zeros(n, dtype=bool)
    rotulo_ord = np.zeros(n, dtype=float)

    bin_td = pd.Timedelta(hours=HORAS_POR_BIN[freq])
    fim_serie = tempos[-1]
    # Ultima leitura de cada dia; um bloco curto nao pode terminar depois dela.
    ultima_do_dia = pd.Series(tempos).groupby(tempos.normalize()).max()

    limiar = (_como_timedelta(limiar_gap_real)
              if limiar_gap_real is not None else None)
    lacunas_reais = []
    if limiar is not None:
        difs = pd.Series(tempos).diff()
        for i in np.flatnonzero((difs > limiar).to_numpy()):
            lacunas_reais.append((tempos[i - 1], tempos[i]))

    for valor in _como_lista(gaps):
        duracao = _como_timedelta(valor)
        if duracao <= pd.Timedelta(0):
            raise ValueError(f"gaps: duracao nao positiva: {valor!r}")
        if duracao < bin_td:
            continue          # duracao menor que um bin: nao cabe na granularidade

        # Inicios que ainda cabem inteiros na serie observada.
        cabem = np.flatnonzero((tempos + duracao) <= fim_serie)
        if not len(cabem):
            continue

        for _tentativa in range(50):
            j = int(cabem[rng.integers(len(cabem))])
            inicio = tempos[j]
            fim = inicio + duracao
            if duracao < LIMIAR_GAP_NOTURNO:
                ultima = ultima_do_dia.get(inicio.normalize())
                if ultima is None or fim > ultima:
                    continue  # bloco curto entraria na pausa noturna
            if any(inicio < real_fim and fim > real_inicio
                   for real_inicio, real_fim in lacunas_reais):
                continue      # bloco cruzaria um gap real
            k = int(tempos.searchsorted(fim, side="left"))
            lo = max(0, j - 1)
            hi = min(n, k + 1)
            if not mascara_ord[lo:hi].any():
                break         # bloco livre, com 1 linha de respiro dos demais
        else:
            continue          # sem posicao livre: nao sobrepoe bloco nem gap real

        mascara_ord[j:k] = True
        rotulo_ord[j:k] = duracao / pd.Timedelta(hours=1)

    mascara = np.zeros(n, dtype=bool)
    rotulo = np.zeros(n, dtype=float)
    mascara[ordem] = mascara_ord
    rotulo[ordem] = rotulo_ord

    return mascara, rotulo
