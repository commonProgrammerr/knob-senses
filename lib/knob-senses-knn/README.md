# knob-senses-knn

k-NN (k = 3 vizinhos, 1000 prototipos) de regressao de **PCA** (potencia CA do
inversor, em kW) do projeto knob-senses, gerado com [emlearn](https://emlearn.readthedocs.io).

Gerado por `utils/treino_knn.ipynb`. Nao edite os arquivos de `src/` a mao: a proxima
execucao do notebook os sobrescreve.

## Uso

```cpp
#include <knob_senses_knn.h>

const float bruto[KNOB_KNN_N_FEATURES] = { 480.0f, 310.5f, 26.1f };

const float kw = knn_pca_prever(bruto);
```

## Contrato

A ordem de `bruto` e' **temperatura_ambiente, ghit, jacques, sintetico**. **Passe valores BRUTOS** (grandeza fisica): o
z-score acontece dentro da lib, com as constantes de `normalizacao.json` - se o dado
chegar em z-score, a conta acontece duas vezes e a previsao colapsa num valor fixo.

O emlearn 0.23.2 impoe quantizacao int16 (z: `lrintf(z*1000.0)` com clamp em
+-8.0; alvo em miliwatts) e nao tem regressao - a MEDIA dos k vizinhos e' da
lib. A previsao sai em passos de 1 mW, ceifada em `[0.01, 2.02]`.

## Procedencia

| | |
|---|---|
| prototipos | 1000 (subconjunto deterministico do treino, semente do contrato) |
| k | 3 |
| MAE no teste | 0.1476 kW |
| R2 no teste | 0.7669 |

Medido no split de **teste**, que e' futuro de verdade (corte por tempo, sem embaralhar).

## Custo

| | |
|---|---|
| flash (ESP32-S3) | ~9.8 KB (dados + rotulos int16, `static const`) |
| RAM | ~7.8 KB (scratch do emlearn: 8 B por prototipo) |

A RAM e' o preco de usar o `eml_neighbors_infer` como vem (ele exige o array de
distancias completo). A varredura do notebook escolhe o numero de prototipos trocando
acuracia por flash E RAM.
