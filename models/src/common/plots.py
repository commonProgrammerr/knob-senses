"""Gráficos compartilhados dos notebooks.

A paleta e o `enfeitar` estavam copiados em quatro notebooks; a serie temporal em dois;
a importância de features e os hexbin em dois. Este modulo concentra essas copias.

matplotlib NAO e' dependência base: ele esta no extra `notebooks`. `import common` nao
puxa este modulo; so' quem plota importa `common.plots`.
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


def plot_time_series(dataset: pd.DataFrame, janela: int = 7):
    """Serie diária de PCA por inversor, um painel por par de anos.

    `dataset` precisa de `datetime`, `inversor` e `PCA`; PCA == 0 é descartado.
    """
    d = dataset[dataset.PCA != 0].copy()
    d["ano"] = d["datetime"].dt.year
    d["dia"] = d["datetime"].dt.normalize()
    anos, estacoes = sorted(d["ano"].unique()), sorted(d["inversor"].unique())
    cores = dict(zip(estacoes, plt.cm.tab10(np.linspace(0, 1, len(estacoes)))))

    # pares montados DE TRAS PARA FRENTE: com contagem impar, o ano mais antigo
    # fica sozinho e o ULTIMO ano sempre tem um par.
    pares, i = [], len(anos)
    while i > 0:
        pares.append(anos[max(i - 2, 0):i])
        i -= 2
    pares.reverse()

    fig, axs = plt.subplots(len(pares), 1, figsize=(14, 4 * len(pares)),
                            squeeze=False, sharey=True)

    for ax, par in zip(axs.ravel(), pares):
        falhas = 0
        for est in estacoes:
            d_est = d[(d["inversor"] == est) &
                      (d["ano"] >= par[0]) & (d["ano"] <= par[-1])]
            if d_est.empty:
                continue

            serie = d_est.groupby("dia")["PCA"].mean().sort_index()
            cor = cores[est]

            ax.plot(serie.index, serie.values, color=cor, lw=0.6, alpha=0.25,
                    marker=".", ms=2)
            rotulo = f"{est} {par[0]}" if len(par) == 1 else f"{est} {par[0]}-{par[-1]}"
            ax.plot(serie.rolling(janela, center=True, min_periods=1).mean(),
                    color=cor, lw=1.8,
                    label=f"{rotulo} ({serie.mean():.2f} kW)")

            esperado = pd.date_range(serie.index.min(), serie.index.max(), freq="D")
            faltando = esperado.difference(serie.index)
            if len(faltando):
                blocos = faltando.to_series().diff().ne(pd.Timedelta(days=1)).cumsum()
                for _, bloco in faltando.to_series().groupby(blocos):
                    ax.axvspan(bloco.iloc[0] - pd.Timedelta(hours=12),
                               bloco.iloc[-1] + pd.Timedelta(hours=12),
                               facecolor=cor, alpha=0.10, zorder=0)
                falhas += len(faltando)

        titulo = str(par[0]) if len(par) == 1 else f"{par[0]} - {par[-1]}"
        if falhas:
            titulo += f"  -  {falhas} dias sem leitura"
        enfeitar(ax, titulo, "data", "PCA médio diário (kW)")
        ax.legend(frameon=False, loc="upper left", fontsize=9,
                  labelcolor=TINTA, ncol=3)

    fig.subplots_adjust(top=0.94, bottom=0.05, left=0.06, right=0.98, hspace=0.18)
    


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
