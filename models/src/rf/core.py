"""O caminho do Random Forest: treino, varredura de tamanho e geracao da lib C.

A lib C entregue (lib/knob-senses-rf/) e' escrita pelo generate_c_lib() deste modulo: as
arvores saem do emlearn como CODIGO (method="inline"), e as constantes de
normalizacao vao compiladas junto — o firmware nao le' JSON nenhum.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from common import contract, data, paths
from common.templates import fill_template

# Escolhidos na varredura (ver treino_random_forest.ipynb, secao 5): e' onde a curva
# dobra — comprar acuracia a partir dali custa flash em proporcao bem pior.
N_TREES, MIN_LEAF = 30, 50
N_JOBS = -1


def train(X, y, n_trees: int = N_TREES, min_leaf: int = MIN_LEAF):
    """A floresta final, ja treinada. Semente fixa: o mesmo dado gera a mesma lib C."""
    return RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=min_leaf,
                                 n_jobs=N_JOBS, random_state=contract.SEED).fit(X, y)


def count_nodes(model) -> int:
    return sum(e.tree_.node_count for e in model.estimators_)


def count_leaves(model) -> int:
    return sum(int((e.tree_.children_left == -1).sum()) for e in model.estimators_)


def tree_depth(model) -> int:
    return max(e.tree_.max_depth for e in model.estimators_)


def sweep(X_train, y_train, X_test, y_test, configs=((300, 2), (100, 2), (50, 5), (50, 20),
                                                     (30, 50), (20, 100), (12, 200), (6, 400))):
    """Tamanho x acuracia: quantas arvores e quantas amostras minimas por folha.

    Devolve a tabela (quem imprime e' o notebook). A coluna `kb` e' estimativa grosseira
    (12 bytes por no' + 4 por folha) — serve para COMPARAR configuracoes, nao para
    prometer flash: o numero real sai da compilacao, secao 9 do notebook.
    """
    rows = []
    for n_trees, min_leaf in configs:
        t0 = time.perf_counter()
        m = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=min_leaf,
                                  n_jobs=N_JOBS, random_state=contract.SEED).fit(X_train, y_train)
        secs = time.perf_counter() - t0
        pred = m.predict(X_test)
        n_nodes, n_leaves = count_nodes(m), count_leaves(m)
        rows.append({
            "arvores": n_trees, "folha": min_leaf, "nos": n_nodes,
            "kb": (n_nodes * 12 + n_leaves * 4) / 1024, "prof": tree_depth(m),
            "mae": mean_absolute_error(y_test, pred), "r2": r2_score(y_test, pred), "s": secs,
        })
    return pd.DataFrame(rows)


HEADER_TEMPLATE = '''/*
 * knob-senses - Random Forest de regressao de PCA (potencia CA do inversor, kW).
 *
 * GERADO POR utils/treino_random_forest.ipynb. NAO EDITE A MAO: a proxima execucao
 * do notebook sobrescreve este arquivo.
 *
 * Como usar:
 *
 *     #include <knob_senses_rf.h>
 *
 *     const float bruto[KNOB_RF_N_FEATURES] = { 26.1f, 480.0f, 38.7f };
 *     const float kw = pca_prever(bruto);
 *
 * ORDEM DAS GRANDEZAS BRUTAS:
@@ORDER_C@@
 *
 * O modelo foi treinado em z-scores e a normalizacao vive DENTRO desta lib (ver
 * knob_senses_rf.cpp): passe a grandeza fisica como ela veio do sensor. Nao normalize
 * antes de chamar pca_prever() - se o dado chegar em z-score, a conta acontece duas
 * vezes, tudo cai a ~7 desvios-padrao da media, e a previsao COLAPSA num valor fixo.
 *
 * Custo no ESP32-S3: 67.6 KB de flash, 0 de RAM. A floresta e' emitida como codigo
 * (emlearn method='inline'), nao como tabela - por isso nao ocupa RAM.
 */
#ifndef KNOB_SENSES_RF_H
#define KNOB_SENSES_RF_H

#include <stddef.h>

/* --------------------------------------------------------------- contrato */

#define KNOB_RF_N_FEATURES @@N_FEATURES@@

/* Ordem esperada em bruto[]. As 4 primeiras sao exatamente a ordem que
 * utils/server.ipynb transmite pela rede, entao o firmware entrega o que
 * recebeu sem reordenar. */
#define KNOB_RF_ORDEM "@@ORDER_LIST@@"

/* Faixa fisica do alvo: o inversor satura em ~2 kW e o dataset nao tem zero (o
 * 0 era sentinela de faltante, removido na analise). A saida ja vem ceifada. */
#define KNOB_RF_PCA_MIN @@PCA_MIN@@f
#define KNOB_RF_PCA_MAX @@PCA_MAX@@f

/* ------------------------------------------------------------ procedencia */

#define KNOB_RF_N_ARVORES    @@N_TREES@@
#define KNOB_RF_FOLHA_MIN    @@MIN_LEAF@@
#define KNOB_RF_N_NOS        @@N_NODES@@
#define KNOB_RF_MAE_TESTE    @@MAE@@f   /* kW, no split de teste (futuro) */
#define KNOB_RF_R2_TESTE     @@R2@@f

#ifdef __cplusplus
extern "C" {
#endif

/* bruto -> z-score -> floresta -> clip. Devolve a potencia CA em kW. */
float pca_prever(const float bruto[KNOB_RF_N_FEATURES]);

#ifdef __cplusplus
}
#endif

#endif /* KNOB_SENSES_RF_H */
'''

IMPLEMENTATION_TEMPLATE = '''#include "knob_senses_rf.h"

/* As arvores, geradas pelo emlearn. Este header tem arrays static e so depende de
 * <stdint.h>; inclua-o em UM unico .cpp, senao a tabela entra duplicada no binario
 * (sem erro de link, mas com flash desperdicado). */
#include "modelo_pca.h"

/* Media e desvio, ajustados SO no split de treino (nunca na validacao nem no
 * teste). Recalculados do bruto a cada execucao de utils/treino_random_forest.ipynb
 * e conferidos contra data/processed/*.csv - ver a secao 2 do notebook. */
static const float PCA_MEDIA[KNOB_RF_N_FEATURES] = {
@@MEAN_C@@
};

static const float PCA_DESVIO[KNOB_RF_N_FEATURES] = {
@@STD_C@@
};

float pca_prever(const float bruto[KNOB_RF_N_FEATURES])
{
    float z[KNOB_RF_N_FEATURES];
    for (size_t i = 0; i < KNOB_RF_N_FEATURES; i++) {
        z[i] = (bruto[i] - PCA_MEDIA[i]) / PCA_DESVIO[i];
    }

    float p = modelo_pca_predict(z, KNOB_RF_N_FEATURES);

    /* A floresta extrapola fora do que viu; a faixa fisica nao. */
    if (p < KNOB_RF_PCA_MIN) p = KNOB_RF_PCA_MIN;
    if (p > KNOB_RF_PCA_MAX) p = KNOB_RF_PCA_MAX;
    return p;
}
'''

README_TEMPLATE = '''# knob-senses-rf

Random Forest de regressao de **PCA** (potencia CA do inversor, em kW) do projeto
knob-senses, gerada com [emlearn](https://emlearn.readthedocs.io).

Gerada por `utils/treino_random_forest.ipynb`. Nao edite os arquivos de `src/` a mao: a
proxima execucao do notebook os sobrescreve.

## Uso

```cpp
#include <knob_senses_rf.h>

const float bruto[KNOB_RF_N_FEATURES] = { 26.1f, 480.0f, 38.7f };

const float kw = pca_prever(bruto);
```

## Contrato

A ordem de `bruto` e' **@@ORDER_LIST@@**.

Essa e' exatamente a ordem que o `utils/model_teste.ipynb` transmite pela rede, entao o
firmware entrega o que recebeu sem reordenar.

**Passe valores BRUTOS.** A normalizacao z-score acontece dentro da lib, com as constantes
de `normalizacao.json` (que fica ao lado deste header).

> **O que da' errado se nao for bruto.** Se o dado chegar **ja em z-score** e a lib normalizar
> de novo, a conta acontece duas vezes: tudo cai a ~7 desvios-padrao da media, todas as
> arvores caem na mesma folha extrema, e a previsao **colapsa num valor fixo**. Se
> `pca_prever()` devolver sempre o mesmo numero com a entrada variando, e' isso.

## O que devolve

Potencia CA em kW, ja ceifada em `[@@PCA_MIN@@, @@PCA_MAX@@]`. O teto e' fisico: o inversor
satura perto de 2 kW. O piso tambem: o dataset de treino e' so diurno e nao contem zero,
entao **a noite a floresta extrapola** e devolve valor nao-nulo - o clip limita o estrago,
nao conserta.

## Procedencia

| | |
|---|---|
| arvores | @@N_TREES@@ |
| folha minima | @@MIN_LEAF@@ |
| nos | @@N_NODES@@ |
| MAE no teste | @@MAE@@ kW |
| R2 no teste | @@R2@@ |

Medido no split de **teste**, que e' futuro de verdade (corte por tempo, sem embaralhar),
entao os numeros sao pessimistas na medida certa.

## Custo

| | |
|---|---|
| flash (ESP32-S3) | 67.6 KB |
| RAM | 0 |

A floresta e' emitida como codigo (`emlearn method='inline'`), nao como tabela em RAM. Em
troca ocupa mais flash que a alternativa (`loadable`, ~39 KB), que exigiria quantizar as
features para `int16_t` e levar dois headers do emlearn para o firmware.
'''

MANIFEST_TEMPLATE = '''{
  "name": "knob-senses-rf",
  "version": "1.0.0",
  "description": "Random Forest de regressao de PCA (potencia CA do inversor, kW) do knob-senses, gerada com emlearn. @@N_TREES@@ arvores, folha minima @@MIN_LEAF@@. Recebe @@N_FEATURES@@ grandezas brutas, normaliza em z-score internamente e devolve kW. 67.6 KB de flash e 0 de RAM no ESP32-S3.",
  "keywords": "random-forest, emlearn, tinyml, machine-learning, regression, arduino, esp32",
  "frameworks": "arduino",
  "platforms": "*",
  "build": {
    "srcDir": "src"
  }
}
'''


def generate_c_lib(model, mean, std, mae, r2, dest_dir: Path = paths.LIB_RF,
                   src_dir: Path = paths.LIB_RF_SRC) -> dict[Path, int]:
    """Escreve a lib do PlatformIO inteira: header, implementacao, README, manifesto,
    normalizacao.json e as arvores (modelo_pca.h, geradas pelo emlearn).

    O notebook do RF chama isto. `mean`/`std` tem que ter saido SO do split de
    treino (data.fit_normalization): sao as constantes que vao compiladas para o
    firmware, e a mesma conta que o pca_prever() faz dentro do MCU.
    """
    import emlearn

    dest_dir, src_dir = Path(dest_dir), Path(src_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)

    c_model = emlearn.convert(model, method="inline", dtype="float")
    c_model.save(file=str(src_dir / "modelo_pca.h"), name="modelo_pca")

    n_nodes = count_nodes(model)
    pca_min, pca_max = data.target_limits()
    values = dict(
        N_FEATURES=len(contract.RF_FEATURES), N_TREES=model.n_estimators, MIN_LEAF=MIN_LEAF,
        N_NODES=n_nodes, MAE=f"{mae:.4f}", R2=f"{r2:.4f}",
        ORDER_C="\n".join(f" *   bruto[{i}] = {c}" for i, c in enumerate(contract.RF_FEATURES)),
        ORDER_LIST=", ".join(contract.RF_FEATURES),
        MEAN_C=",\n".join(f"    {mean[c]:.6f}f" for c in contract.RF_FEATURES),
        STD_C=",\n".join(f"    {std[c]:.6f}f" for c in contract.RF_FEATURES),
        PCA_MIN=pca_min, PCA_MAX=pca_max,
    )

    # As constantes tambem em JSON, para nao precisarem ser transcritas a mao por quem for
    # conferir a normalizacao fora do C. O firmware NAO le este arquivo - ele tem as
    # constantes compiladas em knob_senses_rf.cpp. Isto e' registro, nao configuracao;
    # constante transcrita envelhece sozinha, e a que envelhece errado nao avisa.
    normalization = json.dumps({
        "modelo": "knob-senses-rf",
        "gerado_por": "utils/treino_random_forest.ipynb",
        "features": contract.RF_FEATURES,
        "media": {c: float(mean[c]) for c in contract.RF_FEATURES},
        "desvio": {c: float(std[c]) for c in contract.RF_FEATURES},
        "formula_normalizar": "z = (bruto - media) / desvio",
        "formula_desnormalizar": "bruto = z * desvio + media",
        "ajustado_em": "somente o split de treino",
        "aviso": ("Estas sao as constantes compiladas dentro do C da lib; o firmware NAO le "
                  "este arquivo. Alimente pca_prever() com GRANDEZA FISICA: se o dado chegar "
                  "em z-score, a normalizacao acontece duas vezes e a previsao colapsa num "
                  "valor fixo."),
    }, indent=2, ensure_ascii=False) + "\n"

    files = {
        src_dir / "knob_senses_rf.h": HEADER_TEMPLATE,
        src_dir / "knob_senses_rf.cpp": IMPLEMENTATION_TEMPLATE,
        dest_dir / "README.md": README_TEMPLATE,
        dest_dir / "library.json": MANIFEST_TEMPLATE,
        dest_dir / "normalizacao.json": normalization,
    }

    written = {}
    for path, template_text in files.items():
        # write_text com newline padrao (traduz \n -> CRLF no Windows), como o notebook
        # fazia — e o que da' a identidade byte a byte com o que esta' commitado.
        path.write_text(fill_template(template_text, **values), encoding="utf-8")
        written[path] = path.stat().st_size

    # Confere que nenhum marcador sobrou: um @@TOKEN@@ esquecido viraria C invalido.
    for path in files:
        leftover = [t for t in path.read_text(encoding="utf-8").split("@@")[1::2]]
        assert not leftover, f"{path.name} ficou com marcador: {leftover}"

    return written
