# knob-senses

**Soft-sensor com ML para comparação, barateamento e diagnósticos em usinas fotovoltaicas.**

Em vez de medir a potência CA do inversor com instrumentação dedicada, o projeto **estima**
`PCA` (kW CA) a partir de variáveis meteorológicas e de geometria solar — e olha a usina pela
diferença entre o estimado e o medido. A inferência roda em um **ESP32-S3**, com três caminhos
de modelo no mesmo dataset, escolhidos pelo **env** do PlatformIO:

| | Random Forest (emlearn) | MLP (Edge Impulse) | k-NN (emlearn) |
|---|---|---|---|
| notebook de treino | `utils/treino_random_forest.ipynb` | `utils/treino_regressao.ipynb` + Studio | `utils/treino_knn.ipynb` |
| biblioteca | `lib/knob-senses-rf/` | `lib/knob-senses/` (deploy **v9**) | `lib/knob-senses-knn/` |
| env do firmware | `pio run -e rf` | `-e mlp` | `-e knn` |

---

## Contexto — para que serve

| uso | como o projeto atende |
|---|---|
| **Comparação** | o mesmo dataset e o mesmo split comparam os três modelos (Random Forest × MLP × k-NN) e as estações entre si (o teste é 2024 de todas as 6); o servidor de teste fecha a conta `real × pred × err` |
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
data/splited/test.csv                4 177 linhas  (ifpee1, 2024)
data/splited/test/{estacao}.csv      cópia por estação (SÉRIE COMPLETA, todos os anos)
        │
        │  model/ (lib Python do modelo: contrato, split, treino,
        │          geração e verificação da lib C, servidor de teste)
        │
        ├─ utils/treino_random_forest.ipynb ──► lib/knob-senses-rf/   (regenera a lib C)
        └─ utils/treino_regressao.ipynb     ──► Edge Impulse Studio ──► lib/knob-senses/
                                                (DEFASADO — ver a nota no próprio notebook)
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
`python` do PATH é o embutido do ESP-IDF — não use para os notebooks. Os notebooks declaram o
kernelspec `python3`, cujo `argv[0]` é `"python"` sem caminho absoluto (resolve pelo PATH e cai no
Python do ESP-IDF) — para executá-los de linha de comando, passe o kernel explícito:

```bash
.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=knob-senses utils/dataset_split.ipynb
```

> ⚠️ **O `.venv-tf` não está no working tree** (conferido em 2026-10-11). O `nbconvert` foi
> instalado no `.venv` para re-executar os notebooks do split e do RF.
>
> O kernelspec `python3` **de dentro do `.venv`** foi apontado para o interpretador absoluto
> (`<raiz>/.venv/Scripts/python.exe`) em 2026-10-11 — antes ele era o `python` puro, que resolve
> pelo PATH e cai no Python do ESP-IDF (sem `model` e sem pandas). Reinstalar o ipykernel
> (`python -m ipykernel install`) reverte isso.

A lib `model/` (na raiz — layout plano, `module-root = ""` no `pyproject.toml`, porque o `src/`
do projeto é do firmware C++) é instalada **editável** no `.venv`:

```bash
uv pip install -e . --python .venv/Scripts/python.exe
```

Os **seis notebooks** trazem o mesmo install **na primeira célula**: se `model` não importar no
kernel, ela roda
`pip install -e "<raiz>[notebooks]"` sozinha — as dependências de gráfico e de prep
(`matplotlib`, `statsmodels`, `scipy`, `nbconvert`) estão no extra `notebooks` do
`pyproject.toml`. Numa máquina nova: clonar, abrir o notebook no kernel certo e rodar a primeira
célula. Se `model` já importa, a célula não faz nada além de imprimir o caminho do pacote.

> ⚠️ **Não use `uv sync`.** A `.python-version` do repo (3.12, criada pelo `uv init`) faria o uv
> recriar o `.venv` — que é 3.14 e tem pandas/sklearn/emlearn/ipykernel — e derrubar tudo. O 3.12
> provavelmente é intencional (não há wheel de TensorFlow para 3.14), então a `.python-version` fica
> como está; o jeito certo de instalar é o `uv pip install -e .` acima. Se o `.venv-tf` voltar,
> repetir o install nele.

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

## 3. Os três caminhos de modelo

Cada modelo tem um **env do PlatformIO** (`platformio.ini`), e cada env compila só o seu (ver §4):

| | **Random Forest** | **MLP — Edge Impulse** | **k-NN — emlearn** |
|---|---|---|---|
| env | `pio run -e rf` | `-e mlp` | `-e knn` |
| features | 4 · `temperatura_ambiente, ghit, jacques, sintetico` | janela **1 × 3** · `ghit + jacques + temperatura_ambiente` | as do contrato (`RF_FEATURES`) |
| normalização | **dentro da lib** (z-score com as constantes de `normalizacao.json`) | dentro do impulso (standard scaler compilado em `model_variables.h`) | dentro da lib + quantização int16 (imposta pelo emlearn) |
| custo (ESP32-S3) | **69,1 KB** de flash (70 715 B, xtensa `-O2`), 0 de RAM (árvores como código, `emlearn method='inline'`) | arena TFLite de 2 732 bytes | **11,2 KB** de flash e **7,9 KB** de RAM (k=3, 1 000 protótipos, medidos com o xtensa) |
| métrica (split de teste) | MAE **0,1426 kW** · R² **0,7732** (30 árvores, folha mínima 50, 4 824 nós) | — | MAE **0,1476 kW** · R² **0,7669** |
| ordem dos eixos | `KNOB_RF_ORDEM` em `lib/knob-senses-rf/src/knob_senses_rf.h` | `EI_CLASSIFIER_FUSION_AXES_STRING` | `KNOB_KNN_ORDEM` em `lib/knob-senses-knn/src/knob_senses_knn.h` |

> ⚠️ **Os contratos são diferentes** — conjuntos, ordem e normalização. A ordem errada
> **não dá erro**: dá previsão errada em silêncio. O `utils/model_teste.ipynb` confere a ordem
> dele contra a `normalizacao.json` do modelo (`server.check_axis_order()`) antes de transmitir; o
> firmware, para o RF e o k-NN, entrega o que recebeu **sem reordenar**.

O contrato — a ordem das features, a faixa física do alvo, os limites de sanidade e os cortes do
split — vive **num lugar só**: `model/contract.py`. Os notebooks e o servidor de teste importam
de lá; os headers C e os `normalizacao.json` continuam sendo gerados pelos notebooks de cada
modelo, mas a ordem que os alimenta vem do contrato.

> ⚠️ **Ao CONFERIR os CSVs, compare BYTES — não floats.** Re-executar os notebooks reproduz
> `data/splited/` e `lib/knob-senses-rf/` **byte a byte** (medido em 2026-10-11). Mas o parser
> padrão de CSV do pandas 3.0.6 **não arredonda os floats corretamente**, e não é o inverso do
> `to_csv`: o texto `0.30204502076066225` de `data/processed/ifpee1.csv` lê como
> `0.3020450207606622` (com `float_precision="round_trip"` dá o valor certo), e **4 363 de 16 653**
> valores de `PCA` divergem entre os dois parsers. Consequência: comparar um valor recém-lido com um
> valor lido de arquivo reescrito (`np.array_equal`, `DataFrame.equals`) acusa 1 ULP que **não existe
> no disco**. Para conferir, use `Path.read_bytes()`.

**A lib do RF é gerada, não escrita à mão:** `utils/treino_random_forest.ipynb` reescreve
`lib/knob-senses-rf/src/` (árvores, constantes de normalização, `KNOB_RF_ORDEM`, métricas) via
`model/rf.py`. O `data/modelo_rf/verifica.cpp` é o harness nativo que confere o C gerado contra
o CSV de teste (e `model/rf_verify.py` é quem o compila e mede).

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
No caminho RF: `kAxes = 5`, `kPeriodMs = 25` (40 inferências/s).

> ⚠️ **O caminho MLP não está pronto para ligar hoje.** Pelo header v8 ele fica com
> `kAxes = 4` e `kFrameSize = 48`, mas o `main.cpp` aloca `sample[kAxes]` (4 floats) e passa
> esse buffer para a inferência, que espera 48 — leitura fora do buffer. Além disso o servidor
> transmite os 5 eixos do RF, e o `send_prediction()` serializa só um `float` (no MLP o
> `result` é um struct; o que sairia pela rede seriam os 4 primeiros bytes dele).

### Protocolo (porta 3333)

A sessão inteira vive em **uma única conexão TCP** (um só handshake). Mensagens de tamanho
fixo, little-endian; o número de sequência (`seq`, u32) diz a que linha do `test.csv` a
mensagem pertence e volta ecoado em toda resposta:

| mensagem | sentido | bytes |
|---|---|---|
| pedido de amostra | ESP32 → servidor | `\x01` + `seq` u32 |
| amostra | servidor → ESP32 | `\x01` + `seq` u32 + `kAxes` floats (`float32` LE, **grandeza bruta**) |
| fim do dataset | servidor → ESP32 | `\x02` + `seq` u32 (`seq >= 4177`; o firmware para de pedir) |
| predição | ESP32 → servidor | `\x03` + `seq` u32 + 1 `float32` (**a predição**, kW) |
| ACK | servidor → ESP32 | `\x03` + `seq` u32 (a linha `seq` está no log) |

Fluxo por amostra: pedido → amostra → predição → ACK → `seq++`. **A `seq` só avança com o
ACK**: qualquer falha (conexão caída, escrita/leitura parcial, eco divergente) faz `stop()` e
a **mesma** amostra volta no tick seguinte — nunca se pula amostra, então um ensaio completo
tem exatamente 4177 linhas, sem buracos nem linhas trocadas. Predição repetida (retentativa
com ACK perdido no caminho) é ignorada pelo servidor, e o ACK se repete.

Constantes no topo do arquivo: `SERVER_IP`, `SERVER_PORT`, `WIFI_SSID`, `WIFI_PASSWORD`.
O gancho de aquisição é **`exchange_sample()`** (o `request_sample()` é quem fala com o servidor de
teste): hoje ele pede a amostra pela rede; em campo é ali que entram os sensores da usina (ou
uma fonte de dados meteorológicos) — o resto do pipeline não muda.

### Comportamento do loop

- Periodizado por `kPeriodMs`; atraso de um período inteiro conta em **`overruns`** (não tenta
  recuperar em rajada). Com o servidor fora do ar os `overruns` também crescem: cada tentativa
  de `connect` atrasa o período — é esperado, não é problema do modelo.
- Falha de troca conta em **`dropouts`**: a mesma amostra será **retentada** (a `seq` não
  avança). Os prints de falha/volta são só nas transições de estado, não a cada tick.
- Erro de inferência é **fatal**: para de inferir (insistir vazaria a arena do TFLite a cada
  segundo) — e imprime `ERRO FATAL na inferencia`.
- `heap_livre` (maior bloco contíguo) é impresso junto de cada predição: ficando plano ao longo
  de horas, não há fragmentação nem vazamento.

### Gravar e monitorar

```bash
pio run -e rf -t upload   # -e rf | -e mlp | -e knn — um env por modelo (platformio.ini)
pio device monitor
```

O `platformio.ini` tem um `[env]` base (placa, memória, upload, debug, LDF) herdado por três
envs, um por modelo: cada um define `-DMODEL_*` (o `main.cpp` escolhe o include por ele),
`build_src_filter` e `lib_ignore` — as flags específicas (o Edge Impulse, o `-Wa,-I` do
`.incbin`) ficam no env de quem precisa, e nada se mistura.

---

## 5. Servidor de teste (`utils/model_teste.ipynb`)

É o **banco de comparação estimado × medido**: sobe um socket em `0.0.0.0:3333` e serve as
linhas de `data/splited/test.csv` **endereçadas pelo `seq`** do protocolo (as colunas de
features vêm da tupla `AXES` do notebook, conferida por `assert` contra o `normalizacao.json`
do modelo). Cada predição confirmada vira uma linha de `model_eval.csv`
(`sample, real, pred, err`); `PCA == 0` é tratado como label inválido e essas linhas ficam
fora do erro.

O laço vive em `model/server.py` (`server.serve`). O `accept()` fica num laço externo
(tolera reconexão) e o estado da sessão é o `seq`, não a conexão: conexão que morre no meio
volta para o `accept()` e o cliente reconecta reenviando a mesma `seq`; predição repetida
(ACK perdido no caminho) é ignorada sem relogar. Cada conexão tem timeout de 5 s — silêncio
maior que isso é conexão zumbi (ESP reiniciado sem `FIN`) — e é maior que o timeout de leitura
do firmware (2 s), então quem desiste primeiro é sempre o cliente. O `serve()` retorna quando
o firmware pede uma `seq` além do fim (mensagem de fim); até lá, o `model_eval.csv` já gravado
é o fallback que o painel lê se a placa morrer no meio.

> Limitação conhecida: reiniciar o `serve()` no meio de um ensaio **recomeça o log vazio** e o
> firmware continua da `seq` dele — as linhas anteriores se perdem. Para recomeçar do zero,
> suba o servidor e dê reset no ESP (a `seq` volta a 0; o que já estava logado volta como
> duplicata e não é relogado).

> Correção (2026-10-11): até então o arquivo gravava `real` e `pred` **trocados** (a linha saía na
> ordem do dict, não na do cabeçalho). O painel do notebook nunca foi afetado — ele usa o `reply`
> do `serve()`. O `mae` do cabeçalho, que nunca era gravado, saiu.

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
├── model/                  lib Python do modelo: contrato, split, métricas,
│                           treino do RF, geração/verificação da lib C, servidor
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
│   ├── treino_regressao.ipynb      treino do MLP (exporta p/ Edge Impulse) — DEFASADO
│   ├── model_teste.ipynb   servidor de teste (protocolo da §4) + model_eval.csv — a comparação estimado × medido
│   └── model_eval.csv      log do servidor (untracked; criado quando ele roda)
├── data/
│   ├── raw/                CSVs brutos por estação-ano (5 min)
│   ├── processed/          saída do prep (1 h, por estação)
│   ├── splited/            saída do split (o que os treinos leem)
│   └── modelo_rf/          harness nativo de verificação do C gerado (+ model_eval_teste.csv)
├── docs/superpowers/       specs e planos das mudanças de dados
└── platformio.ini          upload/monitor COM28, debug esp-jtag, flags do EI
```

### Estado

O firmware **compila e linka** nos três envs — medido em 2026-10-12:

| env | RAM | flash |
|---|---|---|
| `rf` | 13,6% (44 456 B) | 11,7% (764 933 B) |
| `mlp` | 14,0% (45 996 B) | 11,5% (753 109 B) |
| `knn` | 16,0% (52 472 B — inclui os ~8 KB do scratch do emlearn) | 10,8% (709 593 B) |

Cada build compila **só o seu modelo** (`build_src_filter` + `lib_ignore`): o SDK do Edge
Impulse inteiro só entra no `-e mlp`, e a separação recortou ~50–60 KB de flash de cada binário
(antes, ~807 KB com os caminhos misturados).

A validação ponta a ponta com a placa é feita rodando o servidor (`model_teste.ipynb`) e
olhando o `model_eval.csv`.
