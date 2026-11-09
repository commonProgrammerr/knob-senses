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

# Padroes do train(). O notebook NAO usa estes valores: treina o VENCEDOR da
# varredura (ver treino_random_forest.ipynb, secao 5).
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


# Varredura de tamanho x acuracia: (arvores, amostras_minimas_por_folha, max_depth).
# max_depth None = arvore sem teto de profundidade. As configuracoes intermediarias
# fecham o vao entre 12/200 e 6/400, onde a varredura antiga nao tinha pontos, e
# (6, 200) testa mais capacidade com o MESMO numero de arvores do vencedor antigo.
CONFIGS_SWEEP = (
    (300, 2, None),
    (100, 2, None),
    (50, 5, None),
    (50, 5, 8),
    (50, 20, None),
    (30, 50, None),
    (20, 100, None),
    (15, 150, None),
    (12, 200, None),
    (10, 300, None),
    (8, 400, None),
    (6, 200, None),
    (6, 400, None),
    (6, 800, None),
)


def sweep(X_train, y_train, X_val, y_val, X_test=None, y_test=None,
          configs=CONFIGS_SWEEP):
    """Tamanho x acuracia: arvores, amostras minimas por folha e teto de profundidade.

    Avalia na VALIDACAO temporal (2024H2), nunca no teste para ESCOLHER: escolher
    o tamanho no teste transforma a metrica final em treino disfarcado. O ranking
    usa `r2_cal` — o R2 da validacao com o offset de calibracao do PROPRIO
    candidato aplicado, porque o modelo entregue e' sempre o calibrado: um
    candidato com vies maior e forma melhor nao pode ser punido por um defeito
    que a entrega corrige. `mae`/`r2` (brutos) ficam na tabela so' para mostrar
    quanto o offset mudou. As colunas `mae_tes`/`r2_tes` sao REPORT ONLY: mostram
    o teto de acuracia que o teste permite, mas nao entram no criterio de
    selecao. `s_inf` e' o tempo de inferencia POR AMOSTRA (predict na validacao),
    medido para o desempate do notebook (menor s_inf entre os 2 maiores r2_cal;
    na ausencia da medicao, cai para a menor quantidade de nos). A coluna `kb`
    e' estimativa grosseira (12 bytes por no' + 4 por folha) — serve para
    COMPARAR configuracoes, nao para prometer flash: o numero real sai da
    compilacao, secao 9 do notebook.
    """
    rows = []
    for n_trees, min_leaf, max_depth in configs:
        t0 = time.perf_counter()
        m = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=min_leaf,
                                  max_depth=max_depth, n_jobs=N_JOBS,
                                  random_state=contract.SEED).fit(X_train, y_train)
        secs = time.perf_counter() - t0
        t_inf = time.perf_counter()
        pred = m.predict(X_val)
        s_inf = (time.perf_counter() - t_inf) / max(len(y_val), 1)   # s por amostra
        offset = float(np.mean(y_val - pred))          # offset do PROPRIO candidato
        pred_cal = pred + offset
        n_nodes, n_leaves = count_nodes(m), count_leaves(m)
        row = {
            "arvores": n_trees, "folha": min_leaf, "prof_max": max_depth,
            "nos": n_nodes,
            "kb": (n_nodes * 12 + n_leaves * 4) / 1024, "prof": tree_depth(m),
            "mae": mean_absolute_error(y_val, pred), "r2": r2_score(y_val, pred),
            "offset": offset,
            "mae_cal": mean_absolute_error(y_val, pred_cal),
            "r2_cal": r2_score(y_val, pred_cal),
            "s": secs, "s_inf": s_inf,
        }
        if X_test is not None and y_test is not None:
            pred_tes = m.predict(X_test) + offset
            row["mae_tes"] = mean_absolute_error(y_test, pred_tes)
            row["r2_tes"] = r2_score(y_test, pred_tes)
        rows.append(row)
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
 * Custo: a floresta e' emitida como codigo (emlearn method='inline'), entao nao
 * ocupa RAM; o flash depende do tamanho da floresta - meca no objeto.
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

/* Faixa fisica do alvo: 0..2 kW. O inversor nao devolve potencia negativa e
 * satura perto de 2 kW; a saida ja vem ceifada. */
#define KNOB_RF_PCA_MIN @@PCA_MIN@@f
#define KNOB_RF_PCA_MAX @@PCA_MAX@@f

/* ------------------------------------------------------------ procedencia */

#define KNOB_RF_N_ARVORES    @@N_TREES@@
#define KNOB_RF_FOLHA_MIN    @@MIN_LEAF@@
#define KNOB_RF_N_NOS        @@N_NODES@@
#define KNOB_RF_FREQ         "@@FREQ@@"   /* granularidade do split de treino */
#define KNOB_RF_MAE_TESTE    @@MAE@@f   /* kW, no split de teste (futuro) */
#define KNOB_RF_R2_TESTE     @@R2@@f

/* Calibracao de vies: somada antes do clip. Foi ajustada na janela de
 * calibracao (2024H2), nunca no teste, para corrigir a deriva de eficiencia
 * 2024->2025. Se o modelo for retreinado, recalcule-a no notebook. */
#define KNOB_RF_OFFSET       @@OFFSET@@f

#ifdef __cplusplus
extern "C" {
#endif

/* bruto -> z-score -> floresta -> offset -> clip. Devolve a potencia CA em kW. */
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

    float p = modelo_pca_predict(z, KNOB_RF_N_FEATURES) + KNOB_RF_OFFSET;

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
satura perto de 2 kW. O piso e' 0: o dataset de treino e' so diurno, entao **a noite a
floresta extrapola** e devolve valor nao-nulo - o clip limita o estrago, nao conserta.

## Procedencia

| | |
|---|---|
| arvores | @@N_TREES@@ |
| folha minima | @@MIN_LEAF@@ |
| nos | @@N_NODES@@ |
| granularidade | @@FREQ@@ |
| MAE no teste | @@MAE@@ kW |
| R2 no teste | @@R2@@ |
| calibracao (vies) | @@OFFSET@@ kW |

Medido no split de **teste**, que e' futuro de verdade (corte por tempo, sem embaralhar),
entao os numeros sao pessimistas na medida certa. O offset de calibracao foi ajustado
em **2024H2**, a janela mais recente antes do teste; ele corrige a deriva de eficiencia
e e' somado dentro de `pca_prever()`, antes do clip.

## Custo

| | |
|---|---|
| flash (ESP32-S3) | medir no objeto |
| RAM | 0 |

A floresta e' emitida como codigo (`emlearn method='inline'`), nao como tabela em RAM.
O flash depende do numero de nos; a celula xtensa do notebook mede no objeto.
'''

MANIFEST_TEMPLATE = '''{
  "name": "knob-senses-rf",
  "version": "1.0.0",
  "description": "Random Forest de regressao de PCA (potencia CA do inversor, kW) do knob-senses, gerada com emlearn. @@N_TREES@@ arvores, folha minima @@MIN_LEAF@@, granularidade @@FREQ@@. Recebe @@N_FEATURES@@ grandezas brutas, normaliza em z-score internamente e devolve kW. Arvores em codigo (0 de RAM); o flash depende do tamanho e deve ser medido no objeto.",
  "keywords": "random-forest, emlearn, tinyml, machine-learning, regression, arduino, esp32",
  "frameworks": "arduino",
  "platforms": "*",
  "build": {
    "srcDir": "src"
  }
}
'''


def generate_c_lib(model, mean, std, mae, r2, offset: float = 0.0,
                   freq: str = "1h",
                   dest_dir: Path = paths.LIB_RF,
                   src_dir: Path = paths.LIB_RF_SRC) -> dict[Path, int]:
    """Escreve a lib do PlatformIO inteira: header, implementacao, README, manifesto,
    normalizacao.json e as arvores (modelo_pca.h, geradas pelo emlearn).

    O notebook do RF chama isto. `mean`/`std` tem que ter saido SO do treino
    (data.fit_normalization): sao as constantes que vao compiladas para o
    firmware, e a mesma conta que o pca_prever() faz dentro do MCU. `offset` e' o
    vies medido na calibracao temporal (2024H2), somado antes do clip; o teste
    nunca entra nessa conta.
    """
    import emlearn

    dest_dir, src_dir = Path(dest_dir), Path(src_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)

    c_model = emlearn.convert(model, method="inline", dtype="float")
    c_model.save(file=str(src_dir / "modelo_pca.h"), name="modelo_pca")

    n_nodes = count_nodes(model)
    pca_min, pca_max = data.physical_target_limits()
    values = dict(
        N_FEATURES=len(contract.RF_FEATURES), N_TREES=model.n_estimators,
        MIN_LEAF=model.min_samples_leaf,
        N_NODES=n_nodes, FREQ=freq, MAE=f"{mae:.4f}", R2=f"{r2:.4f}", OFFSET=f"{offset:.6f}",
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
        "freq": freq,
        "features": contract.RF_FEATURES,
        "media": {c: float(mean[c]) for c in contract.RF_FEATURES},
        "desvio": {c: float(std[c]) for c in contract.RF_FEATURES},
        "formula_normalizar": "z = (bruto - media) / desvio",
        "formula_desnormalizar": "bruto = z * desvio + media",
        "ajustado_em": "somente o treino (2021..2024H1)",
        "calibracao_offset_kw": float(offset),
        "calibracao_ajustada_em": "2024H2 (nunca no teste)",
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
