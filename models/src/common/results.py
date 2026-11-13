"""As saidas dos notebooks em results/: CSV para codigo, Markdown para leitura.

O stdout de um notebook nao e' um arquivo: quem chega depois (uma pessoa ou um
agente) nao consegue reler a tabela. Toda tabela que vale conferir e' gravada
duas vezes, em results/<subpasta>/<nome>.csv e .md: o CSV alimenta codigo, o
Markdown e' a leitura direta, sem precisar abrir o .ipynb.

O `to_markdown` do pandas depende do tabulate, que nao esta no ambiente; o
renderizador abaixo cobre o que os notebooks salvam (escalares e datas).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import paths


def pasta(subpasta: str = "") -> Path:
    """`results/<subpasta>`, criada se faltar."""
    destino = paths.RESULTS / subpasta if subpasta else paths.RESULTS
    destino.mkdir(parents=True, exist_ok=True)
    return destino


def _celula(valor) -> str:
    if valor is None or (not isinstance(valor, (list, tuple, dict)) and pd.isna(valor)):
        return ""
    if isinstance(valor, float):
        return f"{valor:.6g}"
    return str(valor).replace("|", "\\|").replace("\n", " ")


def markdown_tabela(df: pd.DataFrame, index: bool = False) -> str:
    """Tabela GFM a partir de um DataFrame, sem depender de tabulate."""
    d = df.reset_index() if index else df
    colunas = [str(c) for c in d.columns]
    linhas = ["| " + " | ".join(colunas) + " |",
              "|" + "|".join(" --- " for _ in colunas) + "|"]
    for valores in d.itertuples(index=False, name=None):
        linhas.append("| " + " | ".join(_celula(v) for v in valores) + " |")
    return "\n".join(linhas)


def salvar_tabela(df: pd.DataFrame, nome: str, subpasta: str = "",
                  index: bool = False) -> tuple[Path, Path]:
    """Grava `nome`.csv e `nome`.md em results/<subpasta> e devolve os caminhos."""
    destino = pasta(subpasta)
    csv = destino / f"{nome}.csv"
    md = destino / f"{nome}.md"
    df.to_csv(csv, index=index)
    md.write_text(markdown_tabela(df, index=index) + "\n", encoding="utf-8")
    print(f"  {csv.relative_to(paths.ROOT)} | {md.relative_to(paths.ROOT)}", flush=True)
    return csv, md


def salvar_texto(texto: str, nome: str, subpasta: str = "") -> Path:
    """Grava um Markdown livre em results/<subpasta>/<nome>.md."""
    destino = pasta(subpasta) / f"{nome}.md"
    destino.write_text(texto.rstrip() + "\n", encoding="utf-8")
    print(f"  {destino.relative_to(paths.ROOT)}", flush=True)
    return destino


def salvar_json(dados, nome: str, subpasta: str = "") -> Path:
    """Grava um JSON legivel em results/<subpasta>/<nome>.json."""
    destino = pasta(subpasta) / f"{nome}.json"
    destino.write_text(json.dumps(dados, indent=2, ensure_ascii=False, default=str) + "\n",
                       encoding="utf-8")
    print(f"  {destino.relative_to(paths.ROOT)}", flush=True)
    return destino
