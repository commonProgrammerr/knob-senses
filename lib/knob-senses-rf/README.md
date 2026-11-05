# knob-senses-rf

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

A ordem de `bruto` e' **temperatura_ambiente, ghit, jacques**.

Essa e' exatamente a ordem que o `utils/model_teste.ipynb` transmite pela rede, entao o
firmware entrega o que recebeu sem reordenar.

**Passe valores BRUTOS.** A normalizacao z-score acontece dentro da lib, com as constantes
de `normalizacao.json` (que fica ao lado deste header).

> **O que da' errado se nao for bruto.** Se o dado chegar **ja em z-score** e a lib normalizar
> de novo, a conta acontece duas vezes: tudo cai a ~7 desvios-padrao da media, todas as
> arvores caem na mesma folha extrema, e a previsao **colapsa num valor fixo**. Se
> `pca_prever()` devolver sempre o mesmo numero com a entrada variando, e' isso.

## O que devolve

Potencia CA em kW, ja ceifada em `[0.0, 2.0]`. O teto e' fisico: o inversor
satura perto de 2 kW. O piso e' 0: o dataset de treino e' so diurno, entao **a noite a
floresta extrapola** e devolve valor nao-nulo - o clip limita o estrago, nao conserta.

## Procedencia

| | |
|---|---|
| arvores | 12 |
| folha minima | 200 |
| nos | 4532 |
| MAE no teste | 0.1137 kW |
| R2 no teste | 0.9109 |
| calibracao (vies) | -0.099050 kW |

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
