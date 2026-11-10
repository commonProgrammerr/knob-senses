"""Leitura, normalizacao e limites do dataset — a parte Python do contrato."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import contract, paths


def read_csv(p: Path, parse: tuple[str, ...] = (contract.INDEX,)) -> pd.DataFrame:
    """O jeito de ler os CSV do projeto: BOM fora, nomes sem espaco, datas parseadas.

    `parse` lista as colunas de data a converter; o default e' so' `datetime`.
    """
    df = pd.read_csv(p, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    for col in parse:
        df[col] = pd.to_datetime(df[col])
    return df


def show_dataset(dataset: pd.DataFrame, colunas: bool = False) -> pd.DataFrame:
    """Imprime o resumo do pool (linhas/colunas, nulos, zeros) e devolve o df.

    `colunas=True` tambem lista as colunas (menos as de controle de arquivo), como
    fazia a copia do dataset_prep.
    """
    print(f"\npool: {len(dataset)} linhas, {dataset.shape[1]} colunas")
    if colunas:
        print("colunas:", sorted(c for c in dataset.columns
                                 if c not in ("arquivo", "estacao", "ano")))
    print("\nnulos por coluna (so as que tem):")
    nulos = dataset.isna().sum()
    print(nulos[nulos > 0].to_string() if (nulos > 0).any() else "  (nenhum)")
    print("\ncolunas com zero:")
    zeros = dataset.isin([0]).sum()
    print(zeros[zeros > 0].to_string() if (zeros > 0).any() else "  (nenhum)")
    return dataset


def load_pool(data_dir: Path = paths.RAW) -> pd.DataFrame:
    """Todos os CSV de um diretorio empilhados, com a coluna `inversor` (nome do arquivo)."""
    parts = []
    for p in sorted(Path(data_dir).rglob("*.csv")):
        df = read_csv(p)
        df["inversor"] = p.stem
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def load_split(name: str, freq: str | None = None) -> pd.DataFrame:
    """Le data/splits/{nome}.csv (o que os treinos e o servidor consomem).

    `freq` filtra a granularidade (valores de `contract.GRANULARIDADES`); sem
    ele, devolve todas as linhas, como no formato antigo (sem coluna `freq`).
    """
    df = read_csv(paths.SPLITS / f"{name}.csv")
    if freq is not None:
        assert "freq" in df.columns, f"{name}.csv nao tem a coluna freq (rodo o prep?)"
        df = df[df["freq"] == freq].reset_index(drop=True)
    return df


def fit_normalization(train: pd.DataFrame, features=contract.RF_FEATURES):
    """Media e desvio SO do treino.

    Se validacao ou teste entrassem na conta, haveria vazamento do futuro para dentro
    do treino. O guarda 0 -> 1 evita divisao por zero numa coluna constante.
    """
    mean = train[list(features)].mean()
    std = train[list(features)].std(ddof=0).replace(0.0, 1.0)
    return mean, std


def z_score(df: pd.DataFrame, mean, std, features=contract.RF_FEATURES) -> pd.DataFrame:
    """A mesma conta que o pca_prever() faz dentro do MCU: z = (bruto - media) / desvio."""
    return (df[list(features)] - mean) / std


def valid_readings(df: pd.DataFrame) -> pd.DataFrame:
    """Leituras fisicamente validas; saturacao e' limitada a 2 kW.

    Apagao (PCA zero com sol), irradiancia baixa e temperatura absurda entram
    fora. O clipping NAO e' removido: o inversor satura em ~2 kW, entao a
    potencia acima disso vira 2,0 kW para a media do bin refletir a saturacao.
    """
    df = df.copy()
    # df.loc[df["PCA"] > 2.0, "PCA"] = 2.0

    invalido = (
        ((df["PCA"] == 0) & (df["ghit"] > 50))   # apagao com sol
        | (df["ghit"] < 5)                       # noite / irradiancia nula
        | (df["temp_amb"] >= 50)                 # sensor de temperatura
        | (df["PCA"] >= 2.0)
    )
    return df[~invalido]


DOWNSAMPLE_FREQ = "1h"

# O alvo e' potencia media (kW): PCA e ghit tem MEDIA, como as temperaturas.
# Agregar por soma mudaria a grandeza para energia/irradiacao e faria o alvo
# depender de quantas amostras validas sobraram no bin.
DOWNSAMPLE_RULES = {
    "PCA": "mean",
    "ghit": "mean",
    "ioeff": "mean",
    "jacques": "mean",
    "kc": "mean",
    "kt": "mean",
    "pvsyst_cell": "mean",
    "sandia": "mean",
    "sandia_cell": "mean",
    "temp_amb": "mean",
}


def downsample(df: pd.DataFrame, rules: dict = DOWNSAMPLE_RULES,
               freq: str = DOWNSAMPLE_FREQ) -> pd.DataFrame:
    """Agrega o pool no passo do split, preservando a unidade fisica.

    Potencia, irradiancia e temperaturas sao intensivas: media. O alvo vira
    potencia media em kW. Usado pelo prep (que grava os splits) e por
    `dataset_limits` (que precisa do mesmo agregado para conferir a grandeza
    fisica). Nao imprime: quem chama decide se quer o resumo.
    """
    grupo = ["inversor", pd.Grouper(key="datetime", freq=freq)]
    colunas = ["inversor", "datetime", *rules]
    return df.dropna(subset=colunas).groupby(grupo).agg(rules).reset_index()


def infer_freq(df: pd.DataFrame, col: str = contract.INDEX):
    """Passo base da grade da serie, como Timedelta.

    Usa a MENOR diferenca positiva, nao a mediana/moda: um split amostrado por
    linha pode ter a maioria dos intervalos em 2 h mesmo quando o dado e' de 1 h.
    A menor diferenca e' o passo real da grade. Sem duas linhas, cai no
    `DOWNSAMPLE_FREQ`.
    """
    diffs = pd.to_datetime(df[col]).sort_values().diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    return diffs.min() if len(diffs) else pd.Timedelta(DOWNSAMPLE_FREQ)


def _range_limits(df: pd.DataFrame, features, margin: float) -> dict:
    """Min/max de cada feature no df, com folga relativa."""
    limites = {}
    for c in features:
        low, high = float(df[c].min()), float(df[c].max())
        folga = (high - low) * margin if high > low else 1.0
        limites[c] = (low - folga, high + folga)
    return limites


@lru_cache(maxsize=None)
def _pool_limits(features: tuple[str, ...], margin: float, freq) -> dict:
    """Limites do pool fisico, sem olhar para os splits (o check nao pode ser circular).

    Fontes, em ordem: `data/processed` (pool do prep antigo, se existir);
    `data/raw` filtrado por `valid_readings` e agregado por `downsample` (o prep
    atual le raw direto e nao escreve processed — ver utils/dataset_prep.ipynb);
    split de treino, so' como ultimo recurso se nem raw houver. A agregacao usa
    o mesmo passo da grade do split que esta' sendo conferido: `ghit` e `PCA`
    somam no bin, entao comparar um split de 10 min com o raw de 5 min daria
    metade do limite.
    """
    pool = Path(paths.PROCESSED)
    if pool.exists() and any(pool.rglob("*.csv")):
        df = load_pool()
    elif Path(paths.RAW).exists():
        df = load_pool(paths.RAW)
        df = df[df["inversor"].str.startswith("ifpee")]
        df["PCA"] = df["PCA"].fillna(0)   # mesma sentinela de faltante do prep
        df = valid_readings(df)
        df = downsample(df, freq=freq)
    else:
        df = load_split("train")
    return _range_limits(df, features, margin)


def dataset_limits(df: pd.DataFrame | None = None,
                   features=contract.RF_FEATURES, margin: float = 0.10,
                   freq=None) -> dict:
    """Faixa observada de cada feature, com folga relativa, no dataset.

    Sem `df`, usa o pool fisico (ver `_pool_limits`), que inclui o ano de teste:
    o split de teste e' futuro e pode ter ghit maior que qualquer ano do treino,
    entao limitar a sanidade ao treino quebra em dado fisico legitimo. `freq` e'
    o passo da grade do split (ver `infer_freq`); sem ele, usa `DOWNSAMPLE_FREQ`.

    Com `df`, calcula a faixa do proprio df. E' a fonte dos limites de sanidade:
    mudou o dataset, os limites acompanham.
    """
    if df is None:
        return dict(_pool_limits(tuple(features), float(margin),
                                 freq if freq is not None else pd.Timedelta(DOWNSAMPLE_FREQ)))
    return _range_limits(df, features, margin)


def target_limits(df: pd.DataFrame | None = None,
                  margin: float = 0.05) -> tuple[float, float]:
    """Faixa do alvo observada no dataset, com folga relativa.

    Serve para inspecao/estatistica. Para o CLIP de saida use
    `physical_target_limits()`: a folga observada pode permitir valor negativo.
    """
    if df is None:
        df = load_split("train")
    low, high = float(df[contract.TARGET].min()), float(df[contract.TARGET].max())
    folga = (high - low) * margin if high > low else 1.0
    return low - folga, high + folga


def physical_target_limits() -> tuple[float, float]:
    """Faixa fisica do alvo em kW: potencia CA nao negativa, saturacao em 2 kW."""
    return 0.0, 2.0


def check_physical_scale(name: str, df: pd.DataFrame,
                         limites: dict | None = None) -> None:
    """Confere que o CSV esta em GRANDEZA FISICA.

    Os limites saem do pool fisico (`dataset_limits`), com 10% de folga: raw
    filtrado e agregado no MESMO passo da grade do df, ou `data/processed` se ele
    existir. Nao saem dos splits — um check derivado dos proprios splits passaria
    por construcao e nao pegaria o retorno acidental do z-score para dentro dos
    CSV.
    """
    limites = dataset_limits(freq=infer_freq(df)) if limites is None else limites
    faltando = [c for c in limites if c not in df.columns]
    assert not faltando, f"{name}.csv: faltam colunas {faltando}"
    for c, (low, high) in limites.items():
        lowest, highest = float(df[c].min()), float(df[c].max())
        assert low <= lowest and highest <= high, (
            f"{name}.csv: {c} fora de [{low:.4f}, {high:.4f}] "
            f"(min {lowest:.4f}, max {highest:.4f}) - o csv nao esta em grandeza fisica")
