# knob-senses-rf

Random Forest de regressao de **PCA** (potencia CA do inversor, em kW) do projeto
knob-senses, gerada com [emlearn](https://emlearn.readthedocs.io).

Gerada por `utils/treino_random_forest.ipynb`. Nao edite os arquivos de `src/` a mao: a
proxima execucao do notebook os sobrescreve.

## Uso

```cpp
#include <knob_senses_rf.h>

const float bruto[KNOB_RF_N_FEATURES] = {
    26.1f, 480.0f, 310.5f, 38.7f,   // temperatura, ghi_cs, ghit, jacques
    sin1, cos1,                      // do ciclo anual, calculados no device
};

const float kw = pca_prever(bruto);
```

## Contrato

A ordem de `bruto` e' **temperatura_ambiente, ghit, jacques, sin1, cos1, PCA_lag1, sintetico**.

As 4 primeiras sao exatamente a ordem que o `utils/server.ipynb` transmite pela rede, entao o
firmware entrega o que recebeu sem reordenar. `sin1`/`cos1` sao o 1o harmonico do ciclo anual
e tambem chegam pela rede, ja calculados.

**Passe valores BRUTOS.** A normalizacao z-score acontece dentro da lib, com as constantes
de `normalizacao.json` (que fica ao lado deste header).

> **O que da' errado se nao for bruto.** Se o dado chegar **ja em z-score** e a lib normalizar
> de novo, a conta acontece duas vezes: tudo cai a ~7 desvios-padrao da media, todas as
> arvores caem na mesma folha extrema, e a previsao **colapsa num valor fixo**. Se
> `pca_prever()` devolver sempre o mesmo numero com a entrada variando, e' isso.

## O que devolve

Potencia CA em kW, ja ceifada em `[0.01, 2.02]`. O teto e' fisico: o inversor
satura perto de 2 kW. O piso tambem: o dataset de treino e' so diurno e nao contem zero,
entao **a noite a floresta extrapola** e devolve valor nao-nulo - o clip limita o estrago,
nao conserta.

## Procedencia

| | |
|---|---|
| arvores | 30 |
| folha minima | 50 |
| nos | 4698 |
| MAE no teste | 0.1400 kW |
| R2 no teste | 0.7711 |

Medido no split de **teste**, que e' futuro de verdade (corte por tempo, sem embaralhar),
entao os numeros sao pessimistas na medida certa.

## Custo

| | |
|---|---|
| flash (ESP32-S3) | ~93 KB |
| RAM | 0 |

A floresta e' emitida como codigo (`emlearn method='inline'`), nao como tabela em RAM. Em
troca ocupa mais flash que a alternativa (`loadable`, ~39 KB), que exigiria quantizar as
features para `int16_t` e levar dois headers do emlearn para o firmware.
