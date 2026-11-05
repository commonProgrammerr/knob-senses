# knob-senses

**Soft-sensor com ML para comparação, barateamento e diagnósticos em usinas fotovoltaicas.**

Em vez de medir a potência CA do inversor com instrumentação dedicada, o projeto **estima**
`PCA` (kW CA) a partir de variáveis meteorológicas e de geometria solar — e olha a usina pela
diferença entre o estimado e o medido. A inferência roda em um **ESP32-S3**, com três caminhos
de modelo no mesmo dataset, escolhidos pelo **env** do PlatformIO:

| | Random Forest (emlearn) | MLP (Keras / Edge Impulse) | k-NN (emlearn) |
|---|---|---|---|
| notebook de treino | `utils/treino_random_forest.ipynb` | `utils/treino_mlp.ipynb` + Studio | `utils/treino_knn.ipynb` |
| biblioteca | `lib/knob-senses-rf/` | `lib/knob-senses/` (deploy **v9**) | `lib/knob-senses-knn/` |
| env do firmware | `pio run -e rf` | `-e mlp` | `-e knn` |
| estado atual | lib gerada; MAE **0,0681 kW** · R² **0,9688** | modelo local; MAE **0,0604 kW** · R² **0,9770** | ⚠ **desatualizado** — não usar como referência |

---

## Contexto — para que serve

| uso | como o projeto atende |
|---|---|
| **Comparação** | o mesmo dataset e o mesmo split comparam os três modelos (Random Forest × MLP × k-NN). O teste é **2025 de `ifpee1`**; as cópias `data/splits/test/{inversor}.csv` permitem comparar as três usinas no mesmo ano. O servidor de teste fecha a conta `real × pred × err` |
| **Barateamento** | a instrumentação por usina cai para sensores meteorológicos/geométricos + um ESP32-S3: acompanhar a geração de cada inversor deixa de exigir medidor de potência AC e telemetria dedicada |
| **Diagnóstico** | desvio sistemático entre estimado e medido é o sinal de falha (sujeira, sombreamento, degradação). A análise de eficiência (`utils/dataset_analysis.ipynb`, `PCA/ghit` por mês) mostra a perda a partir de 2024 contra a média de 2021–2023 |

O `data/raw/` tem **6 usinas** (`ifpec1-3`, `ifpee1-3`), mas o `dataset_prep.ipynb` seleciona
apenas `ifpee*.csv` — **3 usinas**. `ifpee1` é a única com histórico longo (**2021–2025**);
`ifpee2` e `ifpee3` só têm **2024–2025**. Treino e validação usam apenas `ifpee1`; o teste é o
último ano (2025) de `ifpee1`, e as cópias por inversor repetem 2025 das três usinas.

A série é **só diurna** (filtro `angzen <= 80` na origem). `PCA == 0` é **sentinela de
faltante, não medição**: o prep preenche os nulos com `0` antes dos filtros, e o que sobra de
zero é tratado como label inválido pelo servidor de teste.

---

## 1. A cadeia

```
data/raw/ifpee*_*.csv                 5 min, só diurno (filtro angzen <= 80 na origem)
        │
        │  utils/dataset_prep.ipynb     filtros + downsample 1 h + split, num notebook só
        ▼
data/splits/train.csv                 14 089 linhas  (ifpee1, 2021-01-04..2024-06-30)
data/splits/calibration.csv            2 023 linhas  (ifpee1, 2024H2; calibração do viés)
data/splits/test.csv                   4 039 linhas  (ifpee1, 2025)
data/splits/test/{inversor}.csv        cópia de 2025 por inversor (ifpee1/2/3)
        │
        │  models/src/ (common, rf, knn, mlp: contrato, dados, métricas, geração e
        │              verificação da lib C, servidor de teste)
        │
        ├─ utils/treino_random_forest.ipynb ──► lib/knob-senses-rf/   (regenera a lib C)
        ├─ utils/treino_knn.ipynb           ──► lib/knob-senses-knn/  (regenera a lib C)
        └─ utils/treino_mlp.ipynb           ──► dist/modelo.keras
                                                └─► Edge Impulse Studio ──► lib/knob-senses/
        │
        ▼
src/main.cpp (ESP32-S3)  ◄── WiFi :3333 ──  utils/model_teste.ipynb (servidor de teste)
```

> `data/processed/` **não é mais produzido** — o prep antigo que gerava um CSV por estação foi
> substituído pelo pipeline que escreve direto em `data/splits/`. A pasta `data/splited/`
> existe vazia no repositório antigo; não é usada. `data/modelo_exportado/` também é resíduo
> de uma execução antiga do MLP; o destino atual é `dist/`.

### Ambientes Python

| venv | Python | tem | para quê |
|---|---|---|---|
| `.venv` | 3.13.13 | pandas 3.1.0, scikit-learn, statsmodels, emlearn 0.23.2, TensorFlow 2.21.0, Keras 3.15.1, nbconvert | prep, análise, treino e exportação |

Não há mais `.venv-tf`: o TensorFlow/Keras está no `.venv` principal. O kernelspec
`knob-senses` aponta para `<raiz>/.venv/Scripts/python.exe`; o kernelspec `knob-senses-tf`
ficou registrado no Jupyter apontando para um `.venv-tf` que não existe — **use o kernel
`knob-senses`**.

O `python` do PATH é o embutido do ESP-IDF — não use para os notebooks. Os notebooks declaram
o kernelspec `python3`, cujo `argv[0]` é `"python"` sem caminho absoluto (resolve pelo PATH e
cai no Python do ESP-IDF). Para executá-los de linha de comando, passe o kernel explícito:

```bash
.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=knob-senses utils/dataset_prep.ipynb
```

Os módulos Python vivem em `models/src/{common,rf,knn,mlp}/` e são instalados **editáveis** no
`.venv`:

```bash
uv pip install -e . --python .venv/Scripts/python.exe
```

O `pyproject.toml` usa **hatchling** e uma única distribuição (`knob-sense-models`) com quatro
módulos top-level. `uv_build` não serve: ele exige `src/<nome_do_projeto>/` e não empacota
vários módulos top-level. Todas as dependências (TensorFlow/Keras, matplotlib, statsmodels,
emlearn, setuptools) estão no bloco `dependencies` — não há extras. O `setuptools` entra por
causa do emlearn 0.23.2, que ainda importa `distutils` no Python ≥ 3.12.

Os notebooks **também** põem `models/src` no `sys.path` na primeira célula; o editable é o
caminho suportado, mas o import funciona sem ele quando o notebook roda de `utils/`.

> ⚠️ **Não use `uv sync`.** Ele recria o `.venv` e derruba as dependências instaladas. O jeito
> certo de instalar é o `uv pip install -e .` acima.

---

## 2. Pipeline de dados

### `dataset_prep.ipynb` — trata, agrega e separa

O prep e o split moram no **mesmo notebook** (o antigo `dataset_split.ipynb` foi incorporado).

- Junta os arquivos `data/raw/ifpee*.csv` (9 arquivos: `ifpee1` 2021–2025, `ifpee2/3`
  2024–2025; 410 451 linhas brutas).
- Descarta `condicao`, `velocidade_vento` e `timestamp`; descarta linhas com `PCA` nulo.
- Filtra, antes de agregar:
  - **outage** — `(PCA == 0) & (ghit > 50)`;
  - **saturação** — `PCA > 2,0` vira 2,0 kW (não é removido; a média do bin reflete o clipping);
  - **baixa irradiância** — `ghit < 5`;
  - **temperatura alta** — `temperatura_ambiente >= 50`.
- **Downsample para 1 h** com regras explícitas:
  - `PCA: mean` e `ghit: mean` — **potência média (kW)** e irradiância média (W/m²); o alvo
    não é a soma horária;
  - `ioeff`, `jacques`, `kc`, `kt`, `pvsyst_cell`, `sandia`, `sandia_cell`,
    `temperatura_ambiente`: `mean` (temperaturas e índices são intensivos).
  O downsample descarta linhas com `NaN` e agrega; depois os zeros (`PCA < 0,001`) saem
  (são 150 zeros no pool horário de 36 211 linhas). O Box-Cox exige alvo positivo.
- **Split temporal**, sem embaralhar:
  - `train` = 2021 até 2024-06-30 (2024H1);
  - `calibration` = 2024H2 (2024-07-01..2024-12-31), a janela onde o viés é medido;
  - `test` = 2025 (último ano de `ifpee1`).
- Saída: `data/splits/{train,calibration,test}.csv` + `data/splits/test/{inversor}.csv`
  (2025 de cada usina).

O CSV de split tem **12 colunas**:

```
inversor, datetime, PCA,
ghit, ioeff, jacques, kc, kt, pvsyst_cell, sandia, sandia_cell, temperatura_ambiente
```

O treino usa apenas as **3 features do contrato** (`temperatura_ambiente`, `ghit`, `jacques`);
as demais ficam disponíveis para análise. **Não há lag**: o split não cria `PCA_lag1` nem
colunas sintéticas; o alvo é o `PCA` da mesma hora.

### `dataset_analysis.ipynb` — EDA

Não participa do pipeline de treino. Serve para inspecionar a série, a distribuição por hora
do dia, os buracos de leitura e a eficiência `PCA/ghit` por mês (o diagnóstico de degradação).

---

## 3. Os três caminhos de modelo

Cada modelo tem um **env do PlatformIO** (`platformio.ini`), e cada env compila só o seu (ver
§4):

| | **Random Forest** | **MLP — Keras / Edge Impulse** | **k-NN — emlearn** |
|---|---|---|---|
| env | `pio run -e rf` | `-e mlp` | `-e knn` |
| features | 3 · `temperatura_ambiente, ghit, jacques` | 3 · mesma ordem no modelo local | 3 · mesma ordem |
| normalização | **z-score dentro da lib** (constantes de `normalizacao.json`) | **min-max** ajustado no treino; o impulso v9 do Studio tem **standard scaler próprio** | z-score + quantização int16 dentro da lib |
| custo (ESP32-S3) | ~2 KB estimados (188 nós; medir no objeto), 0 de RAM | arena TFLite de **2 732 bytes** | ~7,8 KB de flash e ~7,8 KB de RAM (1 000 protótipos, k=1) — números da lib antiga |
| métrica (teste 2025) | MAE **0,0681 kW** · RMSE **0,0876 kW** · R² **0,9688** · viés **+0,0079 kW** | MAE **0,0604 kW** · RMSE **0,0753 kW** · R² **0,9770** · viés **+0,0032 kW** | ⚠ MAE **112,9661 kW** · R² **−19,0515** — lib desatualizada |
| ordem dos eixos | `KNOB_RF_ORDEM` em `lib/knob-senses-rf/src/knob_senses_rf.h` | `EI_CLASSIFIER_FUSION_AXES_STRING` | `KNOB_KNN_ORDEM` em `lib/knob-senses-knn/src/knob_senses_knn.h` |

A linha de base (prever a média do treino) tem MAE **0,4291 kW** e R² ≈ **0** no mesmo teste;
é o número a bater para dizer que o modelo aprendeu algo.

> ⚠️ **Os contratos são diferentes — conjuntos, ordem e normalização.** A ordem errada **não
> dá erro**: dá previsão errada em silêncio. O `server.check_axis_order()` existe para conferir
> `AXES` contra a `normalizacao.json` da lib do RF, mas **o notebook atual não o chama** —
> rode-o à mão antes de transmitir. O firmware, para o RF e o k-NN, entrega o que recebeu
> **sem reordenar**. O caminho MLP não é coberto por esse check: o impulso v9 tem outra ordem
> e outro scaler (ver §6).

O contrato vive **num lugar só**: `models/src/common/contract.py`.

```python
RF_FEATURES = ("temperatura_ambiente", "ghit", "jacques")   # ordem de entrada
TARGET = "PCA"
INDEX = "datetime"
SEED = 42
TRAIN_INVERSOR = "ifpee1"
```

`RF_FEATURES` é a ordem que o modelo e o servidor usam. Não há lag. Os **limites de
sanidade** e o **clip do alvo** são dinâmicos:

- `data.dataset_limits()` usa `data/processed` se ele existir; como hoje não existe, agrega
  `data/raw` no mesmo passo do split (`downsample`) e calcula a faixa;
- `data.target_limits()` usa **sempre o treino** (nunca calibração/teste) e aplica 5% de folga;
  é a faixa observada, para inspeção;
- o **clip de saída** usa `data.physical_target_limits()` = `[0, 2]` kW: potência CA não
  negativa e saturação do inversor.

O split temporal (`train`/`calibration`/`test`) é definido no `dataset_prep.ipynb`; o prep é a
fonte da verdade dos cortes.

### Random Forest

`utils/treino_random_forest.ipynb` faz a varredura de tamanho, escolhe a configuração, treina
a floresta final e **gera a lib C** (`lib/knob-senses-rf/src/`) via `models/src/rf/core.py`.
A escolha atual é a **melhor entre as 2 maiores R² na calibração, com desempate pelo menor
tempo**: **6 árvores / folha 400**, 188 nós. No teste 2025: **MAE 0,0681 kW · R² 0,9688**.
O header gerado declara a mesma configuração e o clip físico `[0, 2]` kW.

O custo real de flash/RAM sai da célula xtensa; neste ambiente o toolchain não está instalado,
então a medição foi pulada. O template da lib não fixa mais um tamanho de flash — meça no
objeto com o PlatformIO.

O que a lib garante:

- recebe **grandeza bruta** (nunca z-score) e normaliza internamente;
- devolve kW já ceifado na faixa física `[0, 2]` kW;
- é gerada como **código** (`emlearn method='inline'`), não como tabela em RAM.

A verificação é dupla: `models/src/common/verify.py` compila o `knob_senses_rf.cpp` **da
entrega** com o g++ do host e compara com o sklearn; depois compila com o xtensa e mede
flash/RAM. O conferidor nativo fica em `data/modelo_rf/` (scratch gitignored).

### MLP — Keras

`utils/treino_mlp.ipynb` treina a rede local e salva o checkpoint. O módulo
`models/src/mlp/core.py` só exporta — **não prepara dados nem treina**.

- Entrada: 3 features, **min-max ajustado somente no treino** (fórmula no notebook).
- Alvo: **potência média em kW**; **Box-Cox** ajustado só no treino (λ ≈ 0,653). A inversa
  para kW, o clip físico `[0, 2]` e o offset de calibração (2024H2) ficam no
  pós-processamento, em `dist/transformacao_alvo.json`.
- Arquitetura: `Dense(128) → Dense(64) → Dense(32)`, GELU, dropout 0,1, saída linear; bias
  inicializado com a média do alvo transformado.
- Loss: **Huber**; `HUBER_DELTA = 0,25 × IQR` do alvo Box-Cox.
- Callbacks: `EarlyStopping(monitor="val_mae", restore_best_weights=True)` e
  `ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-8)`.
- Hiperparâmetros por granularidade, detectada por `data.infer_freq` (menor passo positivo):

| granularidade | epochs | batch | paciência | learning rate |
|---|---:|---:|---:|---:|
| 1 d | 500 | 32 | 60 | 1e-3 |
| **1 h** (atual) | **300** | **64** | **40** | **5e-4** |
| 5 min | 400 | 512 | 60 | 3e-4 |

Resultado atual no teste: **MAE 0,0604 kW · RMSE 0,0753 kW · R² 0,9770 · viés +0,0032 kW**.

O notebook salva `paths.EXPORT_MLP / "modelo.keras"` — hoje `dist/modelo.keras` — e o
`transformacao_alvo.json` com λ, clip, offset e a ordem do pós-processamento. Antes de avaliar
no teste, recarrega o `.keras` e confere que as predições são idênticas.

O export para o Edge Impulse é feito por `mlp.export(modelo, amostras, minimo, maximo, ...)`
(SavedModel batch 1, zip, `.tflite`, amostra `.npy` e `normalizacao.json` com o min-max do
treino). **Não há célula chamando o export no notebook hoje.** O min-max do modelo local não é
o mesmo scaler do impulso v9 já embarcado (ver §6).

Referências usadas para escolher os hiperparâmetros: `docs/referencias.md` (Bengio, Masters &
Luschi, Kingma & Ba, Prechelt, Huber e trabalhos de previsão solar).

### k-NN — emlearn (⚠ desatualizado)

O notebook e a lib estão parados numa execução antiga, com o dataset anterior. Os próprios
outputs mostram **1 820 / 980 / 354 linhas** (hoje os splits têm 14 089 / 2 023 / 4 039), e a
lib entregue tem:

- `KNOB_KNN_MAE_TESTE = 112,9661 kW` e `KNOB_KNN_R2_TESTE = −19,0515`;
- `KNOB_KNN_PCA_MIN/MAX = [28,6635, 177,9665]` — faixa incompatível com o alvo atual, que vai
  de ~0 a ~2 kW;
- `normalizacao.json` com médias/desvios de outro dataset, enquanto o treino atual é horário
  (`ghit` médio no bin de 1 h).

A célula final do notebook tem dois asserts (`MAE < 0,35` e amplitude > 0,1); com os splits
atuais a varredura será recalculada e os números acima deixam de valer — a lib só volta a ser
uma entrega quando o notebook for reexecutado e regerar `lib/knob-senses-knn/`. O
`models/src/knn/core.py` também está com defaults `K=3, N=1000`, enquanto a lib entregue saiu
com `k=1`; a próxima execução do notebook recalcula pela regra de escolha e alinha os dois.

---

## 4. Firmware (`src/main.cpp`)

O firmware é um **cliente WiFi**: ele não lê sensores — pede a amostra ao servidor (§5), roda
a inferência e devolve a predição. O modelo é escolhido pelo **env do PlatformIO**, não por
edição manual de include:

```cpp
#if defined(MODEL_RF)
#include "rf_model.h"
typedef rf_result_t result_t;
#elif defined(MODEL_MLP)
#include "mlp_model.h"
typedef ei_impulse_result_t result_t;
#elif defined(MODEL_KNN)
#include "knn_model.h"
typedef knn_result_t result_t;
#else
#error "Nenhum modelo: use um env do platformio.ini (-e rf | -e mlp | -e knn)"
#endif
```

Cada env define `-DMODEL_*`, seu `build_src_filter` e seu `lib_ignore`; as flags específicas
(o Edge Impulse, o `-Wa,-I` do `.incbin`) ficam no env de quem precisa. Upload, monitor e
debug são comuns.

| env | header | `kSamples` | `kAxes` | `kFrameSize` | `kPeriodMs` | ritmo nominal |
|---|---|---:|---:|---:|---:|---|
| `rf` | `rf_model.h` | 1 | 3 | 3 | 10 ms | 100 inferências/s |
| `knn` | `knn_model.h` | 1 | 3 | 3 | 10 ms | 100 inferências/s |
| `mlp` | `mlp_model.h` | 1 | 3 | 3 | **1 ms** | 1 000 inferências/s nominais; na prática limitado pela rede/servidor |

`kAxes`, `kFrameSize` e `kPeriodMs` vêm dos headers do modelo (`KNOB_RF_N_FEATURES`,
`KNOB_KNN_N_FEATURES` e as macros `EI_CLASSIFIER_*`). Os comentários antigos do
`include/mlp_model.h` ainda dizem 5/9/45/1000; os valores reais do v9 são **1/3/3/1** —
confie nas macros.

### Protocolo (porta 3333)

A sessão inteira vive em **uma única conexão TCP** (um só handshake). Mensagens de tamanho
fixo, little-endian; o número de sequência (`seq`, u32) diz a que linha do `test.csv` a
mensagem pertence e volta ecoado em toda resposta:

| mensagem | sentido | bytes |
|---|---|---|
| pedido de amostra | ESP32 → servidor | `\x01` + `seq` u32 (5 B) |
| amostra | servidor → ESP32 | `\x01` + `seq` u32 + `kAxes` floats (`float32` LE, **grandeza bruta**) |
| fim do dataset | servidor → ESP32 | `\x02` + `seq` u32 (`seq >= 4 039` hoje; o firmware para de pedir) |
| predição | ESP32 → servidor | `\x03` + `seq` u32 + 1 `float32` (**a predição**, kW) |
| ACK | servidor → ESP32 | `\x03` + `seq` u32 (a linha `seq` está no log) |

Fluxo por amostra: pedido → amostra → predição → ACK → `seq++`. **A `seq` só avança com o
ACK**: qualquer falha (conexão caída, escrita/leitura parcial, eco divergente) faz `stop()` e
a **mesma** amostra volta no tick seguinte — nunca se pula amostra, então um ensaio completo
tem exatamente 4 039 linhas, sem buracos nem linhas trocadas. Predição repetida (retentativa
com ACK perdido no caminho) é ignorada pelo servidor, e o ACK se repete.

Constantes no topo de `src/main.cpp`: `SERVER_IP`, `SERVER_PORT`, `WIFI_SSID`,
`WIFI_PASSWORD`. O gancho de aquisição é **`exchange_sample()`** (o `request_sample()` é quem
fala com o servidor de teste): hoje ele pede a amostra pela rede; em campo é ali que entram os
sensores da usina — o resto do pipeline não muda.

### Comportamento do loop

- Periodizado por `kPeriodMs`; atraso de um período inteiro conta em **`overruns`** (não tenta
  recuperar em rajada). Com o servidor fora do ar os `overruns` também crescem: cada tentativa
  de `connect` atrasa o período — é esperado.
- Falha de troca conta em **`dropouts`**: a mesma amostra será **retentada** (a `seq` não
  avança). Os prints de falha/volta são só nas transições de estado, não a cada tick.
- Erro de inferência é **fatal**: para de inferir (insistir vazaria a arena do TFLite a cada
  segundo) — e imprime `ERRO FATAL na inferencia`.
- `heap_livre` (maior bloco contíguo) é impresso junto de cada predição: ficando plano ao
  longo de horas, não há fragmentação nem vazamento.

### Gravar e monitorar

```bash
pio run -e rf -t upload   # -e rf | -e mlp | -e knn — um env por modelo (platformio.ini)
pio device monitor
```

O `platformio.ini` tem um `[env]` base (placa, memória, upload, debug, LDF) herdado por três
envs, um por modelo. `monitor_port`/`upload_port` estão fixos em **COM28** (UART/CH343); o
debug usa o USB nativo (JTAG). Se trocar o cabo de porta, confira com `pio device list`.

---

## 5. Servidor de teste (`utils/model_teste.ipynb`)

É o **banco de comparação estimado × medido**: sobe um socket em `0.0.0.0:3333` e serve as
linhas de **`data/splits/test.csv`** (hoje 4 039 linhas de `ifpee1` em 2025), endereçadas pelo
`seq` do protocolo. As colunas de features vêm de `server.AXES`, que é um alias de
`contract.RF_FEATURES`. O `server.check_axis_order()` compara essa ordem com a
`normalizacao.json` da lib do RF, mas **o notebook atual não o chama**; rode-o manualmente
antes do ensaio.

O laço vive em `models/src/common/server.py` (`server.serve`):

- `accept()` fica num laço externo (tolera reconexão) e o estado da sessão é o `seq`, não a
  conexão;
- conexão que morre no meio volta para o `accept()` e o cliente reconecta reenviando a mesma
  `seq`; predição repetida (ACK perdido no caminho) é ignorada sem relogar;
- cada conexão tem timeout de 5 s — silêncio maior que isso é conexão zumbi (ESP reiniciado
  sem `FIN`) — e é maior que o timeout de leitura do firmware (2 s), então quem desiste
  primeiro é sempre o cliente;
- o `serve()` retorna quando o firmware pede uma `seq` além do fim (mensagem de fim).

O notebook roda um `serve()` por inversor e grava o log em `utils/{inversor}_val.csv` (ex.:
`utils/ifpee1_val.csv`); `paths.EVAL_LOG` (`utils/model_eval.csv`) é o default de `serve()`
quando chamado sem caminho. Cada predição confirmada vira uma linha `sample, real, pred, err`.
`PCA == 0` é label inválido: a linha fica no log, mas fora do erro (hoje são 3 linhas no
teste).

O painel lê o `reply` da sessão ou, se a placa morrer no meio, os `ifpe*_val.csv` já gravados.
Ele mostra a série diária real × predito, o dispersão predito × real colorido pelo erro e a
tabela de métricas por mês, usando `server.eval_metrics`:

| métrica | definição |
|---|---|
| `r2` | coeficiente de determinação |
| `mae` | média de `abs(pred - real)` |
| `rmse` | raiz da média dos quadrados |
| `mape` | média de `abs(erro)/abs(real)` em % |
| `p90` | percentil 90 do erro absoluto |
| `vies` | média de `pred - real` |
| `teto` / `no_teto` | maior predição e quantas linhas encostam nela |

> Limitação conhecida: reiniciar o `serve()` no meio de um ensaio **recomeça o log vazio** e o
> firmware continua da `seq` dele — as linhas anteriores se perdem. Para recomeçar do zero,
> suba o servidor e dê reset no ESP (a `seq` volta a 0; o que já estava logado volta como
> duplicata e não é relogado).

> O print `dados em ...data/processed` da célula de configuração é resquício do pipeline
> antigo; a célula seguinte carrega `data.load_split("test")`, que lê `data/splits/test.csv`.

O comentário sobre `SO_EXCLUSIVEADDRUSE` no notebook importa: no Windows, `SO_REUSEADDR`
permitiria **dois servidores na mesma porta** em silêncio — cada cliente caindo em um.

---

## 6. Apêndice — caminho Edge Impulse (v9)

Projeto `1086855`, deploy **v9**, lib em `lib/knob-senses/` (SDK + modelo embutido, sem rede
nem Python em runtime). O `library.json` dessa lib ainda descreve a **v5** ("5x9", "int8",
"deploy version 5") — ele não acompanha o re-export. A verdade sobre o modelo está em
`src/model-parameters/model_metadata.h`.

### O que é o v9

| Propriedade | Valor |
|---|---|
| Tipo | Regressão (1 saída, `classification[0].value`) |
| Janela | **1 amostra × 3 eixos = 3 floats** (`EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE`) |
| Eixos | `ghit + jacques + temperatura_ambiente` (`EI_CLASSIFIER_FUSION_AXES_STRING`) |
| Intervalo | `EI_CLASSIFIER_INTERVAL_MS` = **1 ms** |
| Arena TFLite | **2 732 bytes** |
| Quantização | entrada e saída **int8** |
| Normalização | **standard scaler** embutido (`EI_CLASSIFIER_HAS_DATA_NORMALIZATION = 1`) |
| Anomalia | não tem (`EI_ANOMALY_TYPE_UNKNOWN`) |

O scaler em `model_variables.h` tem comprimento 3 (não 48): média
`{470,1185, 39,0871, 26,8873}` e escala `{0,003470, 0,090084, 0,244926}`, na ordem dos eixos
**ghit, jacques, temperatura_ambiente**.

> ⚠️ **O impulso v9 não está alinhado com o modelo local atual.** O MLP do notebook treina
> com **min-max** e na ordem `temperatura_ambiente, ghit, jacques`; o impulso embarcado
> normaliza com outro método e em outra ordem, e a média de `ghit` do scaler (470) não bate
> com a média do treino atual (5 425,79). Antes de confiar no `-e mlp` para um ensaio de
> verdade, é preciso reexportar/retreinar o impulso no Studio com os dados e o contrato
> atuais, e conferir eixos + scaler. O firmware entrega o frame bruto; quem normaliza é o
> SDK do Edge Impulse.

### Configuração de build (as duas primeiras não são óbvias)

```ini
lib_ldf_mode = deep+                 ; include por ARQUIVO, não por nome de lib
build_flags = -DEI_PORTING_ESPRESSIF=0
```

E em `lib/knob-senses/library.json`: `"frameworks": "arduino"` (sem isso, o PlatformIO fareja
mbed pela árvore de arquivos e **descarta a biblioteca em silêncio**). O export novo do Edge
Impulse embute o `.tflite` via asm `.incbin`; o driver do gcc não repassa `-I` ao assembler,
por isso o env `mlp` leva `-Wa,-I,lib/knob-senses/src`.

### Regra de ouro do include

`#include <edge-impulse-sdk/classifier/ei_run_classifier.h>` em **exatamente um `.cpp`** do
projeto (o SDK é header-only com definições **não-`inline`**; dois `.cpp` → dezenas de
`multiple definition`). Hoje quem inclui é o `src/mlp_model.cpp`; o `include/mlp_model.h`
inclui só `ei_classifier_types.h`, que traz os tipos sem as implementações.

### Fluxo mínimo

```cpp
float frame[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];   // 3 floats
signal_t signal;
numpy::signal_from_buffer(frame, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
ei_impulse_result_t result = { 0 };
run_classifier(&signal, &result, false);
// result.classification[0].value
```

### Modo contínuo

Não use: no v9, `EI_CLASSIFIER_RAW_SAMPLE_COUNT = 1` e
`EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW = 4`, então `SLICE_SIZE = 1 / 4 = 0` (divisão inteira).
O caminho suportado é uma chamada de `run_classifier()` por amostra.

### Erros comuns

| Código | Significado |
|---|---|
| `EI_IMPULSE_OK` (0) | Sucesso |
| `-1` / `-24` | Formato/tamanho não bate com a entrada |
| `-5` | Erro no bloco de DSP |
| `-3` / `-22` | Erro na inferência TFLite |
| `-6` / `-8` / `-11` | Falha de alocação (heap fragmentado, não falta de RAM — a arena tem 2,7 KB) |
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
| **Predição colapsa num valor fixo** | dado chegou em z-score e a lib normalizou de novo | mande **grandeza bruta** ao RF/k-NN; nunca desnormalize no caminho |
| Previsão variando com viés, sem erro nenhum | ordem de features trocada (falha silenciosa) | conferir `KNOB_RF_ORDEM`/`KNOB_KNN_ORDEM` × `AXES` do servidor × `FEATURES` do notebook |
| MLP com saída sem sentido | impulso v9 usa outra ordem e outro scaler | §6 — reexportar no Studio e conferir eixos + scaler |
| k-NN com erro absurdo | lib treinada no dataset antigo (`PCA` 28–178, `ghit` médio 528) | retreinar `treino_knn.ipynb` nos splits atuais |
| `PCA == 0` entra como label | `0` é sentinela de faltante | o servidor já exclui; nos treinos, confira a fração de zeros |
| Erro do modelo inflado no teste | `PCA == 0` entrou como label | exclua (o servidor faz isso) |
| Dois servidores na mesma porta | `SO_REUSEADDR` no Windows | `SO_EXCLUSIVEADDRUSE` (já no `server.open_socket`) |
| Diferença de 1 ULP ao comparar CSVs | o parser de CSV do pandas não é o inverso exato do `to_csv` | compare **bytes** (`Path.read_bytes()`), não floats |
| Notebook não importa `common` | kernel `python3` caindo no Python do ESP-IDF | use o kernel `knob-senses` ou `sys.path.insert` da primeira célula |

---

## 8. Mapa de arquivos

```
├── models/src/common/      contrato, dados, caminhos, métricas, servidor, verificação,
│                           plots e templates
├── models/src/rf/          treino do Random Forest e geração da lib C
├── models/src/knn/         k-NN (emlearn): quantização, emulador e lib C
├── models/src/mlp/         MLP (Keras): preparo dos arrays e export para o Edge Impulse
├── models/tests/           checks locais: layout, dados, plots e MLP
├── src/
│   ├── main.cpp            cliente WiFi + seleção do modelo pelo env (-DMODEL_*)
│   ├── rf_model.cpp        wrapper do RF (pca_prever + impressão)
│   ├── knn_model.cpp       wrapper do k-NN (knn_pca_prever + impressão)
│   └── mlp_model.cpp       wrapper do EI (run_classifier + impressão)
├── include/
│   ├── rf_model.h          kSamples/kAxes/kFrameSize/kPeriodMs do RF
│   ├── knn_model.h         constantes do k-NN
│   └── mlp_model.h         constantes derivadas das macros do EI
├── lib/
│   ├── knob-senses/        SDK Edge Impulse + modelo v9 (não editar)
│   ├── knob-senses-rf/     RF gerado pelo notebook (não editar src/)
│   └── knob-senses-knn/    k-NN gerado pelo notebook (não editar src/)
├── utils/
│   ├── dataset_prep.ipynb      bruto -> filtros -> 1 h -> splits (tudo num notebook)
│   ├── dataset_analysis.ipynb  EDA + eficiência (PCA/ghit por mês — o diagnóstico)
│   ├── treino_random_forest.ipynb  treina e GERA a lib do RF
│   ├── treino_knn.ipynb            treina e GERA a lib do k-NN (⚠ output antigo)
│   ├── treino_mlp.ipynb            treina o MLP, salva o `.keras` e avalia no teste
│   └── model_teste.ipynb       servidor de teste + painel + tabela de métricas
├── data/
│   ├── raw/                CSVs brutos por estação-ano (5 min); o prep lê só ifpee*
│   ├── splits/             train/calibration/test + cópias 2025 por inversor
│   ├── splited/            pasta vazia legada — não usar
│   ├── modelo_rf/          harness nativo de verificação do C gerado (scratch)
│   └── modelo_exportado/   pasta vazia legada — o destino atual do MLP é dist/
├── dist/                   export do MLP (SavedModel/zip/tflite/npy/json; gitignored)
├── docs/referencias.md     referências dos hiperparâmetros do MLP
├── docs/superpowers/       specs e planos (gitignored)
└── platformio.ini          um env por modelo; upload/monitor COM28; flags do EI
```

### Estado

Métricas registradas no split de teste atual (2025, `ifpee1`; MAE/RMSE em kW):

| modelo | MAE | RMSE | R² | viés | observação |
|---|---:|---:|---:|---:|---|
| média do treino | 0,4291 | 0,4967 | ≈0 | +0,0243 | linha de base |
| Random Forest | **0,0681** | **0,0876** | **0,9688** | +0,0079 | lib entregue; calibrado em 2024H2 |
| MLP (Keras) | **0,0604** | **0,0753** | **0,9770** | +0,0032 | checkpoint local; Box-Cox + calibração |
| k-NN | 112,9661 | — | −19,0515 | — | ⚠ lib antiga; não usar |

Os artefatos em `.pio/build/` foram gerados em datas e revisões diferentes
(`rf` em 2026-11-03, `mlp` em 2026-10-12, `knn` em 2026-10-20) e não são comparáveis entre
si. Para medir o build atual, rode `pio run -e rf`, `pio run -e mlp` ou `pio run -e knn` e
veja o resumo de RAM/flash do PlatformIO.

### Pendências conhecidas

- **k-NN**: lib e outputs antigos; retreinar `treino_knn.ipynb` e regerar
  `lib/knob-senses-knn/` antes de qualquer comparação.
- **MLP embarcado**: o impulso v9 (`lib/knob-senses/`) tem eixos e scaler diferentes do modelo
  local; reexportar/retreinar no Studio e conferir a ordem antes do ensaio na placa.
- **MLP local**: o notebook salva em `dist/`, mas o output antigo ainda imprime
  `data/modelo_exportado/`; o `mlp.export()` não é chamado por nenhuma célula.
- **RF gerado**: o `README.md`/`library.json` da lib não fixam mais um tamanho de flash; meça
  no objeto com a célula xtensa. O header agora declara 6 árvores/folha 400, igual ao notebook.
- **Servidor**: o print `DATA_SOURCE` diz `data/processed`, mas o notebook carrega
  `data/splits/test.csv`; e o `server.check_axis_order()` não é chamado pelo notebook.
- **Docstrings/comentários**: o `include/mlp_model.h` ainda comenta 5/9/45/1000. Os valores
  reais são 1/3/3/1.
- **Split**: a fonte de verdade é o `dataset_prep.ipynb` (`train`/`calibration`/`test`).
