"""Leitura, defasagem e split do dataset — a parte Python do contrato de dados."""

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


def add_time_features(df: pd.DataFrame, col: str = contract.INDEX) -> pd.DataFrame:
    """Adiciona ano/mes/doy/hora/hora_ano a partir de `col`; devolve o mesmo df.

    `doy` e' limitado a 365 (2024 e' bissexto): o eixo de hora acumulada usa 365 dias
    para todos os anos. `hora_ano = (doy - 1)*24 + hora` e' a mesma conta dos paineis
    de serie dos notebooks.
    """
    d = df[col].dt
    df["ano"] = d.year
    df["mes"] = d.month
    df["doy"] = d.dayofyear
    df.loc[df["doy"] == 366, "doy"] = 365
    df["hora"] = d.hour
    df["hora_ano"] = (df["doy"] - 1) * 24 + df["hora"]
    return df


def load_pool(data_dir: Path = paths.PROCESSED) -> pd.DataFrame:
    """Todos os CSV de data/processed empilhados, com a coluna `station` (nome do arquivo)."""
    parts = []
    for p in sorted(Path(data_dir).rglob("*.csv")):
        df = read_csv(p)
        df["station"] = p.stem
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def add_lag1(df: pd.DataFrame) -> pd.DataFrame:
    """PCA da hora anterior como LAG DE TEMPO REAL, por estacao.

    A grade de 24 h e' reconstruida e as posicoes noturnas valem 0,0 — e' o valor
    verdadeiro (de noite o inversor entrega 0), nao imputacao. Assim as 06:00 o
    PCA_lag1 e' 0,0 (nunca a tarde de ontem), e nao ha NaN nem linha dropada.
    """
    df = df.sort_values(["station", contract.INDEX]).copy()
    for _, g in df.groupby("station"):
        grid = pd.date_range(g[contract.INDEX].min().floor("D"),
                             g[contract.INDEX].max(), freq="h")
        series = g.set_index(contract.INDEX)[contract.TARGET].reindex(grid).fillna(0.0)
        df.loc[g.index, contract.LAG] = series.shift(1).reindex(g[contract.INDEX]).to_numpy()
    return df


def split_dataset(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """O split manual: estacao do treino, corte por ano, colunas do contrato, sem embaralhar."""
    base = (df[df.station == contract.TRAIN_STATION][list(contract.SPLIT_COLUMNS)]
            .sort_values(contract.INDEX).reset_index(drop=True))
    year = base[contract.INDEX].dt.year
    cuts = {}
    for name, (year_from, year_to) in contract.SPLIT_CUTS.items():
        mask = pd.Series(True, index=base.index)
        if year_from is not None:
            mask &= year >= year_from
        if year_to is not None:
            mask &= year <= year_to
        cuts[name] = base[mask].reset_index(drop=True)
    return cuts


def save_splits(cuts: dict[str, pd.DataFrame], df: pd.DataFrame,
                out_dir: Path = paths.SPLITS) -> dict[Path, int]:
    """Escreve train/validation/test.csv e a copia por estacao (serie completa) em splited/."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, frame in cuts.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        written[path] = len(frame)
    test_dir = out_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    for station in df.station.unique():
        path = test_dir / f"{station}.csv"
        df['year'] = df.datetime.dt.year
        df = df[df.year == df.year.max()]
        df[df.station == station].to_csv(path, index=False)
        written[path] = int((df.station == station).sum())
    return written


def load_split(name: str) -> pd.DataFrame:
    """Le data/splited/{nome}.csv (o que os treinos e o servidor consomem)."""
    return read_csv(paths.SPLITS / f"{name}.csv")


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


def check_physical_scale(name: str, df: pd.DataFrame) -> None:
    """Confere que o CSV esta em GRANDEZA FISICA (ver contract.LIMITS).

    Este assert e' o que pega o retorno acidental do z-score para dentro dos CSV: se
    ele disparar, o firmware normalizaria duas vezes e a previsao colapsaria.
    """
    for c, (low, high) in contract.LIMITS.items():
        lowest, highest = float(df[c].min()), float(df[c].max())
        assert low <= lowest and highest <= high, (
            f"{name}.csv: {c} fora de [{low}, {high}] (min {lowest:.2f}, max {highest:.2f})"
            " - o csv nao esta em grandeza fisica")
