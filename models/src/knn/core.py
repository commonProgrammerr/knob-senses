"""O caminho do k-NN: varredura, quantizacao, emulador e geracao da lib C.

A lib C entregue (lib/knob-senses-knn/) e' escrita aqui, no mesmo padrao do RF. O modelo
em si (arrays int16 + struct) sai do emlearn (`neighbors.c_generate_neighbors`); o C da
lib chama o pipeline do emlearn (`eml_neighbors_infer`, distancia euclidiana inteira) e
agrega com a MEDIA dos k rotulos - o emlearn 0.23.2 so' faz voto de classe, regressao e'
nossa.

Diferencas que o emlearn impoe (lidas na fonte da versao instalada):

- features E rotulos sao int16 quantizados: zq = lrintf(z * Z_SCALE), yq = lrintf(kW *
  Y_SCALE);
- clamp do z em +-Z_CLAMP: o `eml_isqrt` do emlearn ENTRA EM LACO INFINITO com
  x >= 2**30 (o `q <<= 2` estoura o int32 e vira 0), e com Z_CLAMP = 8 a soma dos
  quadrados das distancias fica em ~7,7e8, com folga para 2**30;
- empates: o quicksort do emlearn compara so' a distancia (instavel). A lib usa um
  qsort com ordem TOTAL (distancia, indice) - e' o que faz o emulador abaixo ser exato.

`np.rint` e' round-half-to-even, igual ao `lrintf` do C no modo de arredondamento
padrao - e' o que deixa o emulador reproduzir o firmware bit a bit.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

from common import contract, data, paths
from common.templates import fill_template

# Defaults da lib entregue; o notebook recalcula pela regra de escolha (escolhe_config).
# Valores da varredura de 2026-10-12: o menor (N, k) a <= 5 % do melhor MAE.
K = 3
N_PROTOTYPES = 1000

# Quantizacao - CONTRATO com o C (ver os templates abaixo): mudar aqui exige regerar.
Z_CLAMP = 8.0
Z_SCALE = 1000
Y_SCALE = 1000

# Varredura do notebook: k x n de prototipos (o notebook acrescenta o conjunto completo).
SWEEP_K = (1, 3, 5, 9, 15)
SWEEP_N = (1000, 2000, 4000)


def quantiza_z(X, media, desvio) -> np.ndarray:
    """Bruto -> z-score -> int16, com o MESMO clamp e arredondamento do C.

    X: DataFrame com as colunas do contrato (media/desvio alinham por nome) ou array
    posicional. `np.rint` = half-to-even = `lrintf` do C.
    """
    Z = np.asarray((X - media) / desvio, dtype=np.float64)
    return np.rint(np.clip(Z, -Z_CLAMP, Z_CLAMP) * Z_SCALE).astype(np.int16)


def quantiza_alvo(y_kw) -> np.ndarray:
    """kW -> miliwatts int16 (10..2020 no split real; a media volta em passos de 1 mW)."""
    return np.rint(np.asarray(y_kw, dtype=np.float64) * Y_SCALE).astype(np.int16)


def quantiza_z_c(X, media, desvio) -> np.ndarray:
    """A quantizacao da CONSULTA no firmware, replicada em float32.

    O bruto chega em float32 (scanf %f), o z-score roda em single e o lrintf arredonda o
    produto float32 - entao o emulador exato tem que fazer a MESMA aritmetica, com as
    constantes como saem impressas no C (%.6f). O quantiza_z() (float64) e' o caminho dos
    DADOS guardados; os dois podem diferir em 1 contagem numa fronteira de arredondamento,
    e os arrays do flash sao o artifact - a consulta e' que precisa bater bit a bit.
    """
    nomes = list(contract.RF_FEATURES)
    X = np.asarray(X, dtype=np.float64)
    med = np.array([float(f"{media[c]:.6f}") for c in nomes], dtype=np.float32)
    des = np.array([float(f"{desvio[c]:.6f}") for c in nomes], dtype=np.float32)
    z = (X.astype(np.float32) - med) / des
    z = np.clip(z, np.float32(-Z_CLAMP), np.float32(Z_CLAMP))
    return np.rint(z * np.float32(Z_SCALE)).astype(np.int16)


def _distancias(zq, Xq) -> np.ndarray:
    """As distancias inteiras do emlearn para UMA consulta (`eml_distance_euclidean_int16`).

    floor da raiz do somatorio, com a correcao de 1 ULP do sqrt em float (perto de
    quadrados perfeitos o float pode voltar um a menos).
    """
    d = np.asarray(Xq, dtype=np.int64) - np.asarray(zq, dtype=np.int64)
    soma = (d * d).sum(axis=1)
    r = np.floor(np.sqrt(soma.astype(np.float64))).astype(np.int64)
    return r + ((r + 1) ** 2 <= soma)


def preve_emulado_lote(Xq, Yq, Zq, k) -> np.ndarray:
    """A previsao do C, replicada em lote - a referencia EXATA da verificacao.

    Usa as distancias inteiras (_distancias) e a ordem TOTAL (distancia, indice) da
    lib; o `argsort` estavel empata pela ordem de indice, que e' exatamente a regra do
    comparador do C.
    """
    Xq = np.asarray(Xq, dtype=np.int64)
    Yq = np.asarray(Yq, dtype=np.int64)
    saida = np.empty(len(Zq), dtype=np.int64)
    for i, z in enumerate(np.asarray(Zq, dtype=np.int64)):
        ordem = np.argsort(_distancias(z, Xq), kind="stable")
        sel = Yq[ordem[:k]]
        saida[i] = (2 * sel.sum() + k) // (2 * k)
    return saida


def preve_emulado(Xq, Yq, zq, k) -> int:
    """O emulador para UMA amostra (conveniencia; ver preve_emulado_lote)."""
    return int(preve_emulado_lote(Xq, Yq, np.asarray([zq]), k)[0])


def _prototipos(n_total: int, n: int) -> np.ndarray:
    """Subconjunto deterministico do treino (semente do contrato): mesma semente, mesma lib."""
    if n >= n_total:
        return np.arange(n_total)
    rng = np.random.default_rng(contract.SEED)
    return np.sort(rng.choice(n_total, n, replace=False))


def sweep(X_treino_q, y_treino_q, X_teste_q, y_teste_kw, n_opcoes=SWEEP_N,
          k_opcoes=SWEEP_K) -> pd.DataFrame:
    """Tamanho x acuracia do k-NN: quantos prototipos e quantos vizinhos.

    sklearn sobre os dados QUANTIZADOS (rapido; a mesma tabela de distancias do C, so'
    sem a ordem total dos empates) - a exatidao fica com o emulador. A coluna `flash_kb`
    soma dados+rotulos int16; `ram_kb` e' o scratch do emlearn (8 B por prototipo).
    """
    X_treino_q = np.asarray(X_treino_q, dtype=np.float32)
    y_treino_q = np.asarray(y_treino_q, dtype=np.float32)
    X_teste_q = np.asarray(X_teste_q, dtype=np.float32)
    y_teste_kw = np.asarray(y_teste_kw, dtype=np.float64)
    n_features = X_treino_q.shape[1]

    linhas = []
    for n in n_opcoes:
        idx = _prototipos(len(X_treino_q), n)
        for k in k_opcoes:
            m = KNeighborsRegressor(n_neighbors=k, metric="euclidean", weights="uniform",
                                    algorithm="brute").fit(X_treino_q[idx], y_treino_q[idx])
            pred = m.predict(X_teste_q) / Y_SCALE
            linhas.append({
                "prototipos": int(len(idx)), "k": k,
                "mae": float(np.abs(pred - y_teste_kw).mean()),
                "r2": float(r2_score(y_teste_kw, pred)),
                "flash_kb": len(idx) * (n_features * 2 + 2) / 1024,
                "ram_kb": 8 * len(idx) / 1024,
            })
    return pd.DataFrame(linhas)


def escolhe_config(tabela: pd.DataFrame) -> tuple[int, int, str]:
    """A regra do spec: ordena por (prototipos, k) crescente; vence a PRIMEIRA cujo MAE
    fica a <= 1,05x do melhor da varredura (ordenacao total - nunca ambigua).

    Devolve (k, prototipos, porque) - o notebook imprime o porque.
    """
    melhor = float(tabela["mae"].min())
    ordenada = tabela.sort_values(["prototipos", "k"], kind="stable")
    linha = ordenada[ordenada["mae"] <= 1.05 * melhor].iloc[0]
    porque = (f"MAE {linha['mae']:.4f} kW (melhor da varredura: {melhor:.4f}; limite de "
              f"5 %: {1.05 * melhor:.4f}) com o menor conjunto/k que atende")
    return int(linha["k"]), int(linha["prototipos"]), porque


def _fecho_headers_emlearn(inc: Path, primeira: str) -> list[str]:
    """O fecho transitivo de `#include` a partir de eml_neighbors.h.

    Os headers do emlearn se referenciam por nome (alguns com <>); so' os que existem
    no pacote entram (stdint.h e companhia ficam de fora).
    """
    import re
    vista: list[str] = []
    fila = [primeira]
    while fila:
        nome = fila.pop(0)
        if nome in vista or not (inc / nome).exists():
            continue
        vista.append(nome)
        texto = (inc / nome).read_text(encoding="utf-8")
        for m in re.finditer(r'#include\s*[<"]([^">]+)[">]', texto):
            fila.append(m.group(1))
    return vista


def _n_linhas(X) -> int:
    return len(X)


def _pega_linhas(X, idx):
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return np.asarray(X)[idx]


KNN_HEADER_TEMPLATE = '''/*
 * knob-senses - k-NN (emlearn) de regressao de PCA (potencia CA do inversor, kW).
 *
 * GERADO POR utils/treino_knn.ipynb. NAO EDITE A MAO: a proxima execucao do notebook
 * sobrescreve este arquivo.
 *
 * Como usar:
 *
 *     #include <knob_senses_knn.h>
 *
 *     const float bruto[KNOB_KNN_N_FEATURES] = { 26.1f, 480.0f, 38.7f };
 *     const float kw = knn_pca_prever(bruto);
 *
 * ORDEM DAS GRANDEZAS BRUTAS:
@@ORDER_C@@
 *
 * O z-score vive DENTRO desta lib e tudo e' quantizado em int16 (imposicao do emlearn):
 * z vira lrintf(z * @@Z_SCALE@@) com clamp em +-@@Z_CLAMP@@f (o clamp garante que o
 * `eml_isqrt` do emlearn nao trave), e o alvo vira miliwatts. A saida e' a MEDIA
 * inteira dos @@K@@ vizinhos mais proximos, de volta para kW e ceifada em
 * [@@PCA_MIN@@, @@PCA_MAX@@].
 *
 * Custo no ESP32-S3: ~@@FLASH_KB@@ KB de flash (dados + rotulos int16) e ~@@RAM_KB@@ KB
 * de RAM (scratch do emlearn: 8 B por prototipo) - o numero de prototipos e' o botao.
 */
#ifndef KNOB_SENSES_KNN_H
#define KNOB_SENSES_KNN_H

#include <stddef.h>

/* --------------------------------------------------------------- contrato */

#define KNOB_KNN_N_FEATURES @@N_FEATURES@@

/* Ordem esperada em bruto[] - a mesma que o servidor de teste transmite. */
#define KNOB_KNN_ORDEM "@@ORDER_LIST@@"

/* Faixa fisica do alvo (a saida ja' vem ceifada). */
#define KNOB_KNN_PCA_MIN @@PCA_MIN@@f
#define KNOB_KNN_PCA_MAX @@PCA_MAX@@f

/* ------------------------------------------------------------- k-NN + quantizacao */

#define KNOB_KNN_N_ITEMS @@N_ITEMS@@   /* prototipos guardados (flash E RAM = 8 x isto) */
#define KNOB_KNN_K @@K@@               /* vizinhos da media */
#define KNOB_KNN_Z_CLAMP @@Z_CLAMP@@f
#define KNOB_KNN_Z_SCALE @@Z_SCALE@@f
#define KNOB_KNN_Y_SCALE @@Y_SCALE@@f

/* ------------------------------------------------------------ procedencia */

#define KNOB_KNN_MAE_TESTE @@MAE@@f   /* kW, no split de teste (futuro) */
#define KNOB_KNN_R2_TESTE @@R2@@f

#ifdef __cplusplus
extern "C" {
#endif

/* bruto -> z/clamp/int16 -> kNN do emlearn -> media de k -> clip. kW. */
float knn_pca_prever(const float bruto[KNOB_KNN_N_FEATURES]);

#ifdef __cplusplus
}
#endif

#endif /* KNOB_SENSES_KNN_H */
'''

KNN_IMPL_TEMPLATE = '''#include "knob_senses_knn.h"

#include <math.h>
#include <stdlib.h>

/* O UNICO TU com o emlearn de vizinhos. ATENCAO dupla: as funcoes do eml_neighbors.h
 * sao globais (incluir em dois TUs duplicaria no LINK), e o proprio header NAO tem
 * include-guard (incluir duas vezes no MESMO TU redefine tudo). O knn_model.h gerado
 * pelo emlearn ja' faz o unico `#include <eml_neighbors.h>` - nao incluir por fora. */
#include "knn_model.h"

/* Media e desvio SO do split de treino (nunca validacao nem teste): as mesmas
 * constantes que o quantiza_z() do notebook usou para gerar os dados. */
static const float KNN_MEDIA[KNOB_KNN_N_FEATURES] = {
@@MEAN_C@@
};

static const float KNN_DESVIO[KNOB_KNN_N_FEATURES] = {
@@STD_C@@
};

/* Scratch da busca: 8 B por prototipo - E' o custo de RAM desta lib (o
 * `eml_neighbors_infer` exige o array de distancias completo). */
static EmlNeighborsDistanceItem s_distancias[KNOB_KNN_N_ITEMS];

/* Ordem TOTAL (distancia, depois indice). O quicksort do emlearn compara so' a
 * distancia - instavel nos empates; com um comparador total, QUALQUER algoritmo de
 * ordenacao da o mesmo resultado, e o emulador da verificacao reproduz. */
static int compara_distancia(const void *pa, const void *pb)
{
    const EmlNeighborsDistanceItem *a = (const EmlNeighborsDistanceItem *)pa;
    const EmlNeighborsDistanceItem *b = (const EmlNeighborsDistanceItem *)pb;
    if (a->distance != b->distance) {
        return (a->distance < b->distance) ? -1 : 1;
    }
    if (a->index != b->index) {
        return (a->index < b->index) ? -1 : 1;
    }
    return 0;
}

/* bruto -> z -> clamp -> int16. O clamp nao e' estetica: com ele a soma dos quadrados
 * das distancias fica < 2**30, que e' onde o `eml_isqrt` do emlearn travaria (o
 * `q <<= 2` estoura o int32 e o laco nunca termina). */
static int16_t quantiza_z(float z)
{
    if (z > KNOB_KNN_Z_CLAMP) {
        z = KNOB_KNN_Z_CLAMP;
    }
    if (z < -KNOB_KNN_Z_CLAMP) {
        z = -KNOB_KNN_Z_CLAMP;
    }
    return (int16_t)lrintf(z * KNOB_KNN_Z_SCALE);
}

float knn_pca_prever(const float bruto[KNOB_KNN_N_FEATURES])
{
    int16_t zq[KNOB_KNN_N_FEATURES];
    for (size_t i = 0; i < (size_t)KNOB_KNN_N_FEATURES; i++) {
        zq[i] = quantiza_z((bruto[i] - KNN_MEDIA[i]) / KNN_DESVIO[i]);
    }

    eml_neighbors_infer(&knn_modelo, zq, KNOB_KNN_N_FEATURES, s_distancias,
                        KNOB_KNN_N_ITEMS);
    qsort(s_distancias, KNOB_KNN_N_ITEMS, sizeof(EmlNeighborsDistanceItem),
          compara_distancia);

    /* A MEDIA dos k rotulos e' nossa: o emlearn 0.23.2 so' tem voto de classe (voto
     * discreto nao serve para regressao de kW). Inteira, com round-half-up: float nao
     * seria reproduzivel bit a bit pelo emulador da verificacao. */
    int32_t soma = 0;
    for (int i = 0; i < KNOB_KNN_K; i++) {
        soma += knn_modelo.labels[s_distancias[i].index];
    }
    const int32_t pred_mw = (2 * soma + KNOB_KNN_K) / (2 * KNOB_KNN_K);

    float kw = pred_mw / KNOB_KNN_Y_SCALE;
    if (kw < KNOB_KNN_PCA_MIN) {
        kw = KNOB_KNN_PCA_MIN;
    }
    if (kw > KNOB_KNN_PCA_MAX) {
        kw = KNOB_KNN_PCA_MAX;
    }
    return kw;
}
'''

KNN_README_TEMPLATE = '''# knob-senses-knn

k-NN (k = @@K@@ vizinhos, @@N_ITEMS@@ prototipos) de regressao de **PCA** (potencia CA do
inversor, em kW) do projeto knob-senses, gerado com [emlearn](https://emlearn.readthedocs.io).

Gerado por `utils/treino_knn.ipynb`. Nao edite os arquivos de `src/` a mao: a proxima
execucao do notebook os sobrescreve.

## Uso

```cpp
#include <knob_senses_knn.h>

const float bruto[KNOB_KNN_N_FEATURES] = { 26.1f, 480.0f, 38.7f };

const float kw = knn_pca_prever(bruto);
```

## Contrato

A ordem de `bruto` e' **@@ORDER_LIST@@**. **Passe valores BRUTOS** (grandeza fisica): o
z-score acontece dentro da lib, com as constantes de `normalizacao.json` - se o dado
chegar em z-score, a conta acontece duas vezes e a previsao colapsa num valor fixo.

O emlearn 0.23.2 impoe quantizacao int16 (z: `lrintf(z*@@Z_SCALE@@)` com clamp em
+-@@Z_CLAMP@@; alvo em miliwatts) e nao tem regressao - a MEDIA dos k vizinhos e' da
lib. A previsao sai em passos de 1 mW, ceifada em `[@@PCA_MIN@@, @@PCA_MAX@@]`.

## Procedencia

| | |
|---|---|
| prototipos | @@N_ITEMS@@ (subconjunto deterministico do treino, semente do contrato) |
| k | @@K@@ |
| MAE no teste | @@MAE@@ kW |
| R2 no teste | @@R2@@ |

Medido no split de **teste**, que e' futuro de verdade (corte por tempo, sem embaralhar).

## Custo

| | |
|---|---|
| flash (ESP32-S3) | ~@@FLASH_KB@@ KB (dados + rotulos int16, `static const`) |
| RAM | ~@@RAM_KB@@ KB (scratch do emlearn: 8 B por prototipo) |

A RAM e' o preco de usar o `eml_neighbors_infer` como vem (ele exige o array de
distancias completo). A varredura do notebook escolhe o numero de prototipos trocando
acuracia por flash E RAM.
'''

KNN_MANIFEST_TEMPLATE = '''{
  "name": "knob-senses-knn",
  "version": "1.0.0",
  "description": "k-NN de regressao de PCA (potencia CA do inversor, kW) do knob-senses, gerado com emlearn. @@N_ITEMS@@ prototipos, k=@@K@@. Recebe grandezas brutas, quantiza em int16, normaliza internamente e devolve kW. ~@@FLASH_KB@@ KB de flash e ~@@RAM_KB@@ KB de RAM no ESP32-S3.",
  "keywords": "knn, k-nearest-neighbors, emlearn, tinyml, machine-learning, regression, arduino, esp32",
  "frameworks": "arduino",
  "platforms": "*",
  "build": {
    "srcDir": "src"
  }
}
'''


def generate_c_lib(X_treino, y_treino, media, desvio, mae, r2, k: int = K,
                   n: int = N_PROTOTYPES, dest_dir: Path = paths.LIB_KNN,
                   src_dir: Path = paths.LIB_KNN_SRC) -> dict[Path, int]:
    """Escreve a lib do PlatformIO inteira: o modelo C do emlearn (knn_model.h), os
    headers vendorizados do emlearn, o header/implementacao, README, manifesto e o
    normalizacao.json.

    `X_treino` na ordem de contract.RF_FEATURES e em grandeza FISICA (DataFrame com as
    colunas do contrato); `media`/`desvio` SO do split de treino
    (data.fit_normalization) - as mesmas contas que o quantiza_z() faz aqui dentro.
    """
    import emlearn

    dest_dir, src_dir = Path(dest_dir), Path(src_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)

    idx = _prototipos(_n_linhas(X_treino), n)
    Xq = quantiza_z(_pega_linhas(X_treino, idx), media, desvio)
    yq = quantiza_alvo(np.asarray(y_treino, dtype=np.float64)[idx])

    # O artifact do emlearn: arrays int16 `static const` + o struct do modelo. Os
    # headers do emlearn vem vendorizados PLANOS no src/ - o knn_model.h gerado os
    # inclui com <angulos>, que so' resolvem pelo -I do src/ da lib.
    c_model = emlearn.neighbors.c_generate_neighbors(Xq, labels=yq, n_neighbors=k,
                                                     prefix="knn_modelo")
    (src_dir / "knn_model.h").write_text(c_model + "\n", encoding="utf-8")

    inc = Path(emlearn.includedir)
    for nome in _fecho_headers_emlearn(inc, "eml_neighbors.h"):
        (src_dir / nome).write_bytes((inc / nome).read_bytes())

    n_features = len(contract.RF_FEATURES)
    n_itens = len(idx)
    pca_min, pca_max = data.physical_target_limits()
    valores = dict(
        N_FEATURES=n_features,
        ORDER_C="\n".join(f" *   bruto[{i}] = {c}" for i, c in enumerate(contract.RF_FEATURES)),
        ORDER_LIST=", ".join(contract.RF_FEATURES),
        PCA_MIN=pca_min, PCA_MAX=pca_max,
        N_ITEMS=n_itens, K=k,
        Z_CLAMP=f"{Z_CLAMP:.1f}", Z_SCALE=f"{Z_SCALE:.1f}", Y_SCALE=f"{Y_SCALE:.1f}",
        MAE=f"{mae:.4f}", R2=f"{r2:.4f}",
        FLASH_KB=f"{n_itens * (n_features * 2 + 2) / 1024:.1f}",
        RAM_KB=f"{8 * n_itens / 1024:.1f}",
        MEAN_C=",\n".join(f"    {media[c]:.6f}f" for c in contract.RF_FEATURES),
        STD_C=",\n".join(f"    {desvio[c]:.6f}f" for c in contract.RF_FEATURES),
    )

    # As constantes tambem em JSON, como no RF - registro, nao configuracao (o firmware
    # tem tudo compilado). As chaves novas (escala_z/clamp_z/escala_y) sao o contrato da
    # quantizacao; `features` continua legivel pelo server.check_axis_order().
    normalization = json.dumps({
        "modelo": "knob-senses-knn",
        "gerado_por": "utils/treino_knn.ipynb",
        "features": contract.RF_FEATURES,
        "media": {c: float(media[c]) for c in contract.RF_FEATURES},
        "desvio": {c: float(desvio[c]) for c in contract.RF_FEATURES},
        "k": k,
        "prototipos": n_itens,
        "escala_z": Z_SCALE,
        "clamp_z": Z_CLAMP,
        "escala_y": Y_SCALE,
        "formula_normalizar": "z = (bruto - media) / desvio; int16: lrintf(clamp(z, +-8.0) * 1000)",
        "formula_desnormalizar": "kW = media_inteira_dos_k_rotulos_em_mW / 1000",
        "ajustado_em": "somente o split de treino",
        "aviso": ("Estas sao as constantes compiladas dentro do C da lib; o firmware NAO le "
                  "este arquivo. Alimente knn_pca_prever() com GRANDEZA FISICA: se o dado "
                  "chegar em z-score, a normalizacao acontece duas vezes e a previsao "
                  "colapsa num valor fixo."),
    }, indent=2, ensure_ascii=False) + "\n"

    files = {
        src_dir / "knob_senses_knn.h": KNN_HEADER_TEMPLATE,
        src_dir / "knob_senses_knn.cpp": KNN_IMPL_TEMPLATE,
        dest_dir / "README.md": KNN_README_TEMPLATE,
        dest_dir / "library.json": KNN_MANIFEST_TEMPLATE,
        dest_dir / "normalizacao.json": normalization,
    }

    written = {}
    for path, template_text in files.items():
        # write_text com newline padrao (traduz \n -> CRLF no Windows), como o RF.
        path.write_text(fill_template(template_text, **valores), encoding="utf-8")
        written[path] = path.stat().st_size

    # Confere que nenhum marcador sobrou: um @@TOKEN@@ esquecido viraria C invalido.
    for path in files:
        leftover = [t for t in path.read_text(encoding="utf-8").split("@@")[1::2]]
        assert not leftover, f"{path.name} ficou com marcador: {leftover}"

    return written
