"""Leitura e split do dataset — a parte Python do contrato de dados."""

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
    df["doy"] = d.day_of_year
    df.loc[df["doy"] == 366, "doy"] = 365
    df["hora"] = d.hour
    df["hora_ano"] = (df["doy"] - 1) * 24 + df["hora"]
    return df


def load_pool(data_dir: Path = paths.PROCESSED) -> pd.DataFrame:
    """Todos os CSV de data/processed empilhados, com a coluna `inversor` (nome do arquivo)."""
    parts = []
    for p in sorted(Path(data_dir).rglob("*.csv")):
        df = read_csv(p)
        df["inversor"] = p.stem
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


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
    for inversor in df.inversor.unique():
        path = test_dir / f"{inversor}.csv"
        df['year'] = df.datetime.dt.year
        df = df[df.year == df.year.max()]
        df[df.inversor == inversor].to_csv(path, index=False)
        written[path] = int((df.inversor == inversor).sum())
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


def fit_min_max(train: pd.DataFrame, features=contract.RF_FEATURES):
    """Minimo e maximo SO do treino (mesma regra do z-score: sem vazamento).

    Se a coluna for constante, o maximo vira `minimo + 1` para a escala nao dividir
    por zero.
    """
    low = train[list(features)].min()
    high = train[list(features)].max()
    return low, high.where(high > low, low + 1.0)


def min_max_scale(df: pd.DataFrame, low, high,
                  features=contract.RF_FEATURES) -> pd.DataFrame:
    """(x - minimo) / (maximo - minimo), com min/max ajustados no treino."""
    return (df[list(features)] - low) / (high - low)


def dataset_limits(df: pd.DataFrame | None = None,
                   features=contract.RF_FEATURES, margin: float = 0.10) -> dict:
    """Faixa observada de cada feature, com folga relativa, no dataset.

    Sem `df`, usa o pool de `data/processed`. E' a fonte dos limites de sanidade:
    mudou o dataset, os limites acompanham.
    """
    if df is None:
        df = load_pool()
    limites = {}
    for c in features:
        low, high = float(df[c].min()), float(df[c].max())
        folga = (high - low) * margin if high > low else 1.0
        limites[c] = (low - folga, high + folga)
    return limites


def target_limits(df: pd.DataFrame | None = None,
                  margin: float = 0.05) -> tuple[float, float]:
    """Faixa do alvo observada no dataset, com folga relativa.

    Sem `df`, usa o split de **treino** (nunca validacao/teste: o clip nao pode
    espiar o futuro). O clip do alvo acompanha o dataset.
    """
    if df is None:
        df = load_split("train")
    low, high = float(df[contract.TARGET].min()), float(df[contract.TARGET].max())
    folga = (high - low) * margin if high > low else 1.0
    return low - folga, high + folga


def check_physical_scale(name: str, df: pd.DataFrame,
                         limites: dict | None = None) -> None:
    """Confere que o CSV esta em GRANDEZA FISICA.

    Os limites saem do dataset de `data/processed` (`dataset_limits`), com 10% de
    folga; nao sao numeros fixos no contrato. Este assert e' o que pega o retorno
    acidental do z-score para dentro dos CSV.
    """
    limites = dataset_limits() if limites is None else limites
    faltando = [c for c in limites if c not in df.columns]
    assert not faltando, f"{name}.csv: faltam colunas {faltando}"
    for c, (low, high) in limites.items():
        lowest, highest = float(df[c].min()), float(df[c].max())
        assert low <= lowest and highest <= high, (
            f"{name}.csv: {c} fora de [{low:.4f}, {high:.4f}] "
            f"(min {lowest:.4f}, max {highest:.4f}) - o csv nao esta em grandeza fisica")
