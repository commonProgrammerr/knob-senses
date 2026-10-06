# knob-senses

**Soft-sensor com ML para comparação, barateamento e diagnósticos em usinas fotovoltaicas.**

Em vez de medir a potência CA do inversor com instrumentação dedicada, o projeto **estima**
`PCA` (kW CA) a partir de variáveis meteorológicas e de geometria solar — e olha a usina pela
diferença entre o estimado e o medido. A inferência roda em um **ESP32-S3**, com dois caminhos
de modelo no mesmo dataset, escolhidos em tempo de compilação:

| | Random Forest (emlearn) — **ativo** | MLP (Edge Impulse) |
|---|---|---|
| notebook de treino | `utils/treino_random_forest.ipynb` | `utils/treino_regressao.ipynb` + Studio |
| biblioteca | `lib/knob-senses-rf/` | `lib/knob-senses/` (deploy **v8**) |
| ativação no firmware | `#include "rf_model.h"` | `#include "mlp_model.h"` |

---

## Contexto — para que serve

| uso | como o projeto atende |
|---|---|
| **Comparação** | o mesmo dataset e o mesmo split comparam os dois caminhos de modelo (Random Forest × MLP) e as estações entre si (o teste é 2024 de todas as 6); o servidor de teste fecha a conta `real × pred × err` |
| **Barateamento** | a instrumentação por usina cai para sensores meteorológicos/geométricos + um ESP32-S3: acompanhar a geração de cada inversor deixa de exigir medidor de potência AC e telemetria dedicada |
| **Diagnóstico** | desvio sistemático entre estimado e medido é o sinal de falha (sujeira, sombreamento, degradação). No próprio repo: a análise de eficiência (`utils/dataset_analysis.ipynb`, `PCA/ghit` por mês) revelou a **perda de 13,9% a partir de ago/2024** contra a média de 2021–2023 |

O dataset cobre 6 estações (`ifpec1-3`, `ifpee1-3`); `ifpee1` é a única com 2021–2024 — as
outras só têm 2024. A série é **só diurna** (filtro `angzen <= 80` na origem) e `PCA == 0` é
**sentinela de faltante, não medição** (em 2021–2023 essas horas foram preenchidas; em 2024
ficaram como `0,0`).

---

## 1. A cadeia

```
data/raw/{estacao}_{ano}.csv         5 min, só diurno (filtro angzen <= 80 na origem)
        │
        │  utils/dataset_prep.ipynb              ~25–50 min (4 ajustes SARIMAX)
        ▼
data/processed/{estacao}.csv         1 h, 2021–2024, + coluna `sintetico`
        │
        │  utils/dataset_split.ipynb             split manual + PCA_lag1
        ▼
data/splited/train.csv               8 311 linhas  (ifpee1, anos < 2023)
data/splited/validation.csv          4 165 linhas  (ifpee1, 2023)
data/splited/test.csv               25 062 linhas  (todas as estações, 2024)
data/splited/test/{estacao}.csv      cópia por estação do teste
        │
        ├─ utils/treino_random_forest.ipynb ──► lib/knob-senses-rf/   (regenera a lib C)
        └─ utils/treino_regressao.ipynb     ──► Edge Impulse Studio ──► lib/knob-senses/
        │
        ▼
src/main.cpp (ESP32-S3)  ◄── WiFi :3333 ──  utils/model_teste.ipynb (servidor de teste)
```

### Ambientes Python

| venv | Python | tem | para quê |
|---|---|---|---|
| `.venv` | 3.14 | pandas, statsmodels, scikit-learn, lightgbm | prep, split, análise |
| `.venv-tf` | 3.13 | TensorFlow/Keras, nbconvert | notebook de treino do MLP |

Kernels Jupyter nomeados: **`knob-senses`** (`.venv`) e **`knob-senses-tf`** (`.venv-tf`). O
`python` do PATH é o embutido do ESP-IDF — não use para os notebooks.

---

## 2. Pipeline de dados

### `dataset_prep.ipynb` — trata e prepara (por estação)

- Junta os arquivos brutos por estação, descarta colunas mortas e faz o **downsample para 1 h**.
- **`PCA == 0` é sentinela de faltante**, não medição: vira `NaN` **no 5-min, antes da média
  horária** (hora com zeros parciais usa só as amostras válidas).
- **Preenche buracos apenas em 2021–2023** (`ifpee1`; as outras estações são só 2024), com
  SARIMAX + bootstrap. São dois tipos:
  - **falta** — hora sem medição (apagão de 20 dias em 2022, 3 dias de jan/2021, horas internas);
  - **zerada** — hora medida (o clima da hora é real) em que o `PCA` era zero no 5-min; só o `PCA` é preenchido.
  284 h preenchidas no total, marcadas na coluna **`sintetico`**.
- **Fora de 2021–2023** (2024) não há síntese: hora sem `PCA` válido é gravada como `0,0`
  (linha preservada).
- Saída: `data/processed/{estacao}.csv` (18 colunas, inclui `sin1`/`cos1` e `sintetico`).

### `dataset_split.ipynb` — o split (manual) + defasagem

- Corte por tempo, sem embaralhar: **train** = `ifpee1` anos < 2023 · **validation** = `ifpee1`
  2023 · **test** = 2024 de todas as estações. `train.csv`/`validation.csv`/`test.csv` +
  `test/{estacao}.csv` saem em `data/splited/`.
- **`PCA_lag1`** = PCA da hora anterior, como **lag de tempo real**: a grade de 24 h é
  reconstruída e as posições noturnas valem `0,0` (é o valor verdadeiro — de noite o inversor
  entrega 0 — não imputação). Consequências: às 06:00 o `PCA_lag1` é `0,0`, nunca a tarde de
  ontem; não há NaN nem linha dropada.
- O lag é calculado **antes** do fatiamento, para cruzar a fronteira treino/validação.

---

## 3. Os dois caminhos de modelo

| | **Random Forest** (ativo) | **MLP — Edge Impulse** |
|---|---|---|
| features | 7 · `temperatura_ambiente, ghit, jacques, sin1, cos1, PCA_lag1, sintetico` | janela **12 × 4** = 48 floats · `temperatura_ambiente, ghi_cs, ghit, jacques` |
| normalização | **dentro da lib** (z-score com as constantes compiladas de `normalizacao.json`) | nenhuma no device — ver §6 |
| custo (ESP32-S3) | ~93 KB de flash, 0 de RAM (árvores como código, `emlearn method='inline'`) | arena TFLite de 3 097 bytes |
| métrica (split de teste) | MAE **0,1400 kW** · R² **0,7711** (30 árvores, folha mínima 50, 4 698 nós) | — |
| ordem dos eixos | `KNOB_RF_ORDEM` em `lib/knob-senses-rf/src/knob_senses_rf.h` | `EI_CLASSIFIER_FUSION_AXES_STRING` |

> ⚠️ **Os dois contratos são diferentes** — conjuntos, ordem e normalização. A ordem errada
> **não dá erro**: dá previsão errada em silêncio. O `utils/model_teste.ipynb` tem um `assert`
> que confere a ordem dele contra a `normalizacao.json` do modelo antes de transmitir; o
> firmware, para o RF, entrega o que recebeu **sem reordenar**.

**A lib do RF é gerada, não escrita à mão:** `utils/treino_random_forest.ipynb` reescreve
`lib/knob-senses-rf/src/` (árvores, constantes de normalização, `KNOB_RF_ORDEM`, métricas).
O `data/modelo_rf/verifica.cpp` é o harness nativo que confere o C gerado contra o CSV de teste.

---

## 4. Firmware (`src/main.cpp`)

O firmware é um **cliente WiFi**: ele não lê sensores — pede a amostra ao servidor (§5),
roda a inferência e devolve a predição. O modelo é escolhido pelo include no topo:

```cpp
#include "rf_model.h"     // Random Forest (ativo)
//#include "mlp_model.h"  // MLP / Edge Impulse
```

O bloco `#if defined(RF_MODEL_H) / #elif defined(MLP_MODEL_H)` liga as funções do modelo
escolhido (`kAxes`, `kFrameSize`, `kPeriodMs`, `run/print/banner` saem do header do modelo).
No caminho RF: `kAxes = 7`, `kPeriodMs = 25` (40 inferências/s).

> ⚠️ **O caminho MLP não está pronto para ligar hoje.** Pelo header v8 ele fica com
> `kAxes = 4` e `kFrameSize = 48`, mas o `main.cpp` aloca `sample[kAxes]` (4 floats) e passa
> esse buffer para a inferência, que espera 48 — leitura fora do buffer. Além disso o servidor
> transmite os 7 eixos do RF, e o `send_inference_results` serializa só um `float` (no MLP o
> `result` é um struct; o que sairia pela rede seriam os 4 primeiros bytes dele).

### Protocolo (porta 3333)

| passo | quem manda | o quê |
|---|---|---|
| 1 | ESP32 → servidor | `\x01` (pedido de amostra) |
| 2 | servidor → ESP32 | `kAxes` floats (`float32` LE, **grandeza bruta**) |
| 3 | ESP32 → servidor | `\x01` (ACK) |
| 4 | ESP32 → servidor | `\x02` + 1 `float32` (**a predição**, kW) |
| — | (cliente) → servidor | `\x00` pula a amostra — existe no servidor, o firmware não usa |

Constantes no topo do arquivo: `SERVER_IP`, `SERVER_PORT`, `WIFI_SSID`, `WIFI_PASSWORD`.
O gancho de aquisição é **`get_sample_data()`**: hoje ele fala com o servidor de teste; em
campo é ali que entram os sensores da usina (ou uma fonte de dados meteorológicos) — o resto
do pipeline não muda.

### Comportamento do loop

- Periodizado por `kPeriodMs`; atraso de um período inteiro conta em **`overruns`** (não tenta
  recuperar em rajada).
- Amostra que não veio conta em **`dropouts`**.
- Erro de inferência é **fatal**: para de inferir (insistir vazaria a arena do TFLite a cada
  segundo) — e imprime `ERRO FATAL na inferencia`.
- `heap_livre` (maior bloco contíguo) é impresso junto de cada predição: ficando plano ao longo
  de horas, não há fragmentação nem vazamento.

### Gravar e monitorar

```bash
pio run -t upload        # esptool, COM28 (platformio.ini)
pio device monitor
```

---

## 5. Servidor de teste (`utils/model_teste.ipynb`)

É o **banco de comparação estimado × medido**: sobe um socket em `0.0.0.0:3333` e percorre as
linhas de `data/splited/test.csv` **na ordem**, uma por iteração (as colunas de features vêm da
tupla `AXES` do notebook, conferida por `assert` contra o `normalizacao.json` do modelo). Cada
predição recebida vira uma linha de `model_eval.csv` (`sample, real, pred, err`); `PCA == 0` é
tratado como label inválido e essas linhas ficam fora do erro.

O comentário sobre `SO_EXCLUSIVEADDRUSE` no notebook importa: no Windows, `SO_REUSEADDR`
permitiria **dois servidores na mesma porta** em silêncio — cada cliente caindo em um.

---

## 6. Apêndice — caminho Edge Impulse (API)

Projeto `1086855`, deploy **v8**, lib em `lib/knob-senses/` (SDK + modelo embutido, sem rede nem
Python em runtime).

### O que é o v8

| Propriedade | Valor |
|---|---|
| Tipo | Regressão (1 saída, `classification[0].value`) |
| Janela | **12 amostras × 4 eixos = 48 floats** (`EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE`) |
| Eixos | `temperatura_ambiente + ghi_cs + ghit + jacques` |
| Intervalo | `EI_CLASSIFIER_INTERVAL_MS` = **300000 ms** (5 min) |
| Arena TFLite | 3 097 bytes |
| Anomalia | não tem (`EI_ANOMALY_TYPE_UNKNOWN`) |

(Os números deste documento valem para a **v8**; a janela desta já foi 5×9 = 45 numa versão anterior.)

### Configuração de build (as duas primeiras não são óbvias)

```ini
lib_ldf_mode = deep+                 ; include por ARQUIVO, não por nome de lib
build_flags = -DEI_PORTING_ESPRESSIF=0
```

E em `lib/knob-senses/library.json`: `"frameworks": "arduino"` (sem isso, o PlatformIO fareja
mbed pela árvore de arquivos e **descarta a biblioteca em silêncio**).

> A `description` desse manifesto ainda fala da **v5** ("5x9", "int8", "deploy version 5") —
> ela não acompanha o re-export. A verdade sobre o modelo está em
> `src/model-parameters/model_metadata.h`.

### Regra de ouro do include

`#include <edge-impulse-sdk/classifier/ei_run_classifier.h>` em **exatamente um `.cpp`** do
projeto (o SDK é header-only com definições **não-`inline`**; dois `.cpp` → dezenas de
`multiple definition`). Hoje quem inclui é o `src/mlp_model.cpp`.

### Fluxo mínimo

```cpp
float frame[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];   // 48 floats, ordem sample-major
signal_t signal;
numpy::signal_from_buffer(frame, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
ei_impulse_result_t result = { 0 };
run_classifier(&signal, &result, false);
// result.classification[0].value
```

### ⚠ O achado do scaler no v8 (conferir no Studio antes de confiar)

No `model-parameters/model_variables.h`, o scaler de entrada tem 48 posições mas **só as 4
primeiras carregam estatística real** — e ela é `média ≈ 0`, `escala ≈ 1`, `var ≈ 1`. As outras
44 são `mean = 0, scale = 1, var = 0` (variância exatamente zero = colunas constantes no treino).
Duas leituras, as duas importantes:

1. **O modelo usa só a primeira amostra da janela** (as 44 posições restantes foram constantes
   no treino) — o mesmo padrão que a v5 tinha.
2. **O scaler é identidade** (`média ≈ 0, variância ≈ 1`): os dados de treino já estavam
   **z-normalizados**, então o modelo **espera z-scores na entrada** — a normalização acontece
   *antes* da rede, não no device. O `mlp_model.cpp` hoje **não** normaliza o frame.

Se for ligar este caminho: alimente z-scores (com as constantes da rodada de treino) e mande as
44 posições restantes como zero, como no treino. O teste decisivo é comparar predições com
entrada bruta × z-score: se só a z-score der valores coerentes, o achado está confirmado.

### Modo contínuo

Agora é divisível: `SLICES_PER_MODEL_WINDOW = 4` e `SLICE_SIZE = 12 / 4 = 3` (na v5 era
`5 / 4 = 1`, e não se aplicava). A API é `run_classifier_init()` + `run_classifier_continuous()`;
nas primeiras `SLICES_PER_MODEL_WINDOW - 1` chamadas a janela ainda está enchendo e a saída não
é válida.

### Erros comuns

| Código | Significado |
|---|---|
| `EI_IMPULSE_OK` (0) | Sucesso |
| `-1` / `-24` | Formato/tamanho não bate com a entrada |
| `-5` | Erro no bloco de DSP |
| `-3` / `-22` | Erro na inferência TFLite |
| `-6` / `-8` / `-11` | Falha de alocação (heap fragmentado, não falta de RAM — a arena tem 3 KB) |
| `-20` | Modelo descontinuado — re-exporte do Studio |
| `-30` | Erro na normalização de dados |

---

## 7. Pegadinhas (e como cada uma se manifesta)

| Sintoma | Causa | Solução |
|---|---|---|
| `No such file or directory` num header que existe | `lib_ldf_mode` no padrão `chain` | `lib_ldf_mode = deep+` |
| A biblioteca some do build, sem aviso | `library.json` sem `"frameworks": "arduino"` | Declare o framework |
| `multiple definition` de dezenas de símbolos | `ei_run_classifier.h` em mais de um `.cpp` | Inclua em um só (wrapper) |
| `multiple definition of 'ei_printf'` | Duas camadas de portabilidade ativas | `-DEI_PORTING_ESPRESSIF=0` |
| **Predição colapsa num valor fixo** | dado chegou em z-score e a lib normalizou de novo | mande **grandeza bruta** ao RF; nunca desnormalize no caminho |
| Previsão variando com viés, sem erro nenhum | ordem de features trocada (falha silenciosa) | conferir `KNOB_RF_ORDEM` × `AXES` do servidor × `FEATURES` do notebook |
| `PCA_lag1` da manhã "errado" | esperava-se o shift posicional (tarde anterior) | é lag de tempo real: às 06:00 vale `0,0` (noite) — por desenho |
| Erro do modelo inflado no teste | `PCA == 0` entrou como label | `0` é sentinela de faltante (só existe em 2024) — exclua |
| Saída do EI sem sentido | entrada fora da ordem/z-score esperados | §6 — conferir o Studio |

---

## 8. Mapa de arquivos

```
├── src/
│   ├── main.cpp            cliente WiFi + seleção do modelo (por include)
│   ├── rf_model.cpp        wrapper do RF (pca_prever + impressão)
│   └── mlp_model.cpp       wrapper do EI (run_classifier + impressão)
├── include/
│   ├── rf_model.h          kAxes/kPeriodMs do caminho RF
│   └── mlp_model.h         constantes derivadas das macros do EI
├── lib/
│   ├── knob-senses/        SDK Edge Impulse + modelo v8 (não editar)
│   └── knob-senses-rf/     RF gerado pelo notebook (não editar src/)
├── utils/
│   ├── dataset_prep.ipynb  bruto -> tratado (por estação)
│   ├── dataset_split.ipynb tratado -> train/validation/test + PCA_lag1
│   ├── dataset_analysis.ipynb  EDA + eficiência (PCA/ghit por mês — o diagnóstico)
│   ├── treino_random_forest.ipynb  treina e GERA a lib do RF
│   ├── treino_regressao.ipynb      treino do MLP (exporta p/ Edge Impulse)
│   └── model_teste.ipynb   servidor de teste (protocolo da §4) + model_eval.csv — a comparação estimado × medido
├── data/
│   ├── raw/                CSVs brutos por estação-ano (5 min)
│   ├── processed/          saída do prep (1 h, por estação)
│   ├── splited/            saída do split (o que os treinos leem)
│   └── modelo_rf/          harness nativo de verificação do C gerado
├── docs/superpowers/       specs e planos das mudanças de dados
└── platformio.ini          upload/monitor COM28, debug esp-jtag, flags do EI
```

### Estado

O firmware **compila e linka** (medido em 2026-10-06: RAM **14,2%** — 46 656 B; Flash **12,3%**
— 807 353 B). Esses 807 KB incluem **os dois caminhos**: o `mlp_model.cpp` é compilado mesmo
quando o `main.cpp` usa o RF, e leva o SDK do Edge Impulse inteiro junto — recortar o caminho
não usado com `build_src_filter` é o ganho de flash mais fácil que existe aqui.

A validação ponta a ponta com a placa é feita rodando o servidor (`model_teste.ipynb`) e
olhando o `model_eval.csv`.
