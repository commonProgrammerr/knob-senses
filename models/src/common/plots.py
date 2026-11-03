"""Gráficos compartilhados dos notebooks.

A paleta e o `enfeitar` estavam copiados em quatro notebooks; a serie temporal em dois;
a importância de features e os hexbin em dois. Este modulo concentra essas copias.

matplotlib e' dependencia do projeto; `import common` nao puxa este modulo; so' quem
plota importa `common.plots`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUPERFICIE, TINTA = "#fcfcfb", "#0b0b0b"
TINTA_2, MUDO = "#52514e", "#898781"
GRADE, EIXO = "#e1e0d9", "#c3c2b7"
REAL, PREDITO = "#2a78d6", "#eb6834"        # slots categóricos 1 e 2
ERRO_CLARO, MAE_ESC = "#86b6ef", "#184f95"  # rampa azul: degraus 250 e 600

HALO = dict(facecolor=SUPERFICIE, edgecolor="none", pad=2.0)


def aplicar_estilo() -> None:
    """Aplica o fundo/fonte do projeto nos rcParams do matplotlib."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "system-ui", "DejaVu Sans"],
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
    })


def enfeitar(ax, titulo, xlabel, ylabel) -> None:
    """Titulo, rótulos, grade e spines no padrão dos notebooks."""
    ax.set_title(titulo, color=TINTA, fontsize=12, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=TINTA_2, fontsize=10)
    ax.set_ylabel(ylabel, color=TINTA_2, fontsize=10)
    ax.grid(True, color=GRADE, linewidth=0.6, linestyle="-")   # solida, nunca tracejada
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(EIXO)
    ax.tick_params(colors=MUDO, labelsize=9)


_ROTULOS_FREQ = {
    "H": "horário", "D": "diário", "W": "semanal",
    "MS": "mensal", "ME": "mensal", "M": "mensal",
    "YS": "anual", "YE": "anual", "Y": "anual",
}


_SUB_DIARIO = frozenset({"H", "T", "MIN", "S", "L", "U", "N"})


def _formata_horas(delta: pd.Timedelta) -> str:
    """Duracao legivel, sempre em horas: 12 h, 480 h, 0.5 h."""
    horas = float(delta.total_seconds()) / 3600
    texto = f"{horas:.2f}".rstrip("0").rstrip(".")
    return f"{texto} h"


def plot_time_series(dataset: pd.DataFrame, janela: int | str = 7, *,
                     freq: str = "D", time_col: str = "datetime",
                     grupo_col: str | None = None, alvo: str = "PCA",
                     unidade: str = "kW", drop_zero: bool = True,
                     sharey: bool = True, ignorar_noite: bool = True,
                     hora_dia: tuple[int, int] = (6, 18),
                     mostrar: bool = True):
    """Série de PCA por grupo, agregada na granularidade pedida.

    `freq` é um alias do pandas: "h" (horário), "D" (diário), "W" (semanal),
    "MS" (mensal), "YS" (anual), etc. `janela` é o número de pontos da média
    móvel (ou um Timedelta, ex. "7D"). Um painel por par de anos.

    Em frequências sub-diárias, `ignorar_noite=True` (default) remove de
    `hora_dia` as horas que não são de leitura; a série é solar. O título do
    painel mostra quantos períodos faltam e a duração total (ex.: "3 períodos
    sem leitura (12 h)", sempre em horas). Com `mostrar=True` (default) chama
    `plt.show()` e devolve `None`; com `mostrar=False` devolve a `Figure`.
    """
    if time_col not in dataset.columns:
        raise ValueError(f"coluna de tempo {time_col!r} ausente")
    if alvo not in dataset.columns:
        raise ValueError(f"coluna alvo {alvo!r} ausente")
    if grupo_col is None:
        grupo_col = next((c for c in ("inversor", "station")
                          if c in dataset.columns), None)
        if grupo_col is None:
            raise ValueError("informe grupo_col: nao achei 'inversor' nem 'station'")

    d = dataset.copy()
    d[time_col] = pd.to_datetime(d[time_col])
    if drop_zero:
        d = d[d[alvo] != 0]
    d["ano"] = d[time_col].dt.year
    anos, grupos = sorted(d["ano"].unique()), sorted(d[grupo_col].unique())
    if not anos or not grupos:
        raise ValueError("dataset sem anos ou grupos para plotar")
    cores = dict(zip(grupos, plt.cm.tab10(np.linspace(0, 1, len(grupos)))))

    # pares DE TRAS PARA FRENTE: com contagem impar, o ano mais antigo fica sozinho.
    pares, i = [], len(anos)
    while i > 0:
        pares.append(anos[max(i - 2, 0):i])
        i -= 2
    pares.reverse()

    fig, axs = plt.subplots(len(pares), 1, figsize=(14, 4 * len(pares)),
                            squeeze=False, sharey=sharey)
    base = freq.upper().lstrip("0123456789").split("-")[0]
    rotulo_freq = _ROTULOS_FREQ.get(base, freq)

    for ax, par in zip(axs.ravel(), pares):
        falhas, duracao = 0, pd.Timedelta(0)
        for grupo in grupos:
            d_g = d[(d[grupo_col] == grupo) &
                    (d["ano"] >= par[0]) & (d["ano"] <= par[-1])]
            if d_g.empty:
                continue

            serie = d_g.set_index(time_col)[alvo].resample(freq).mean().dropna()
            if serie.empty:
                continue
            cor = cores[grupo]
            passo = serie.index.to_series().diff().median()
            if pd.isna(passo) or passo <= pd.Timedelta(0):
                passo = pd.Timedelta(days=1)
            meio = passo / 2

            ax.plot(serie.index, serie.values, color=cor, lw=0.6, alpha=0.22,
                    marker=".", ms=2)
            rotulo = f"{grupo} {par[0]}" if len(par) == 1 else f"{grupo} {par[0]}-{par[-1]}"
            ax.plot(serie.rolling(janela, center=True, min_periods=1).mean(),
                    color=cor, lw=1.8,
                    label=f"{rotulo} ({serie.mean():.2f} {unidade})")

            esperado = pd.date_range(serie.index.min(), serie.index.max(), freq=freq)
            if ignorar_noite and base in _SUB_DIARIO:
                inicio, fim = hora_dia
                esperado = esperado[(esperado.hour >= inicio) & (esperado.hour < fim)]
            faltando = esperado.difference(serie.index)
            if len(faltando):
                blocos = faltando.to_series().diff().ne(passo).cumsum()
                for _, bloco in faltando.to_series().groupby(blocos):
                    ax.axvspan(bloco.iloc[0] - meio, bloco.iloc[-1] + meio,
                               facecolor=cor, alpha=0.10, zorder=0)
                falhas += len(faltando)
                duracao += passo * len(faltando)

        titulo = str(par[0]) if len(par) == 1 else f"{par[0]} - {par[-1]}"
        if falhas:
            nome = "período" if falhas == 1 else "períodos"
            titulo += (f"  -  {falhas} {nome} sem leitura "
                       f"({_formata_horas(duracao)})")
        enfeitar(ax, titulo, "data", f"PCA médio {rotulo_freq} ({unidade})")
        ax.legend(frameon=False, loc="upper left", fontsize=9,
                  labelcolor=TINTA, ncol=3)

    fig.subplots_adjust(top=0.94, bottom=0.05, left=0.06, right=0.98, hspace=0.18)
    if mostrar:
        plt.show()
        return None
    return fig


def plot_importancia(serie: pd.Series, titulo: str = "importancia das features",
                     figsize=(6.5, 4.6), cor: str = "#4c72b0"):
    """Barh horizontal da importancia de features, em ordem crescente."""
    ax = serie.sort_values().plot.barh(figsize=figsize, color=cor)
    ax.set_xlabel("importancia (reducao de impureza)")
    ax.set_title(titulo)
    plt.tight_layout()
    plt.show()
    return ax


def hexbin_real_vs_pred(ax, y, pred, titulo: str | None = None, extent=None,
                        gridsize: int = 45):
    """Hexbin de real x predito, com a diagonal de predicao perfeita."""
    ax.hexbin(y, pred, gridsize=gridsize, bins="log", cmap="viridis", extent=extent)
    if extent is None:
        limite = float(np.nanmax([np.nanmax(y), np.nanmax(pred)])) * 1.06
        extent = (0, limite, 0, limite)
    ax.plot(extent[:2], extent[2:], "r--", lw=1)
    ax.set_xlabel("PCA real (kW)")
    ax.set_ylabel("PCA previsto (kW)")
    if titulo:
        ax.set_title(titulo)
    ax.set_aspect("equal")
    return ax


def hexbin_residuo(ax, y, pred, titulo: str | None = None, gridsize: int = 55):
    """Hexbin do residuo (predito - real) contra o predito."""
    ax.hexbin(pred, pred - y, gridsize=gridsize, bins="log", cmap="magma")
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.set_xlabel("PCA previsto (kW)")
    if titulo:
        ax.set_title(titulo)
    return ax
