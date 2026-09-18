# knob-senses — como usar a API de inferência

Este projeto roda o impulso **knob-senses** (Edge Impulse, projeto `1086855`, deploy **v5**) em um
ESP32-S3. A biblioteca em `lib/knob-senses/` contém o SDK de inferência da Edge Impulse **já com o
modelo embutido** — você não precisa de rede, Python nem TensorFlow em runtime: é só C++ rodando no
microcontrolador.

Este documento ensina a usar a API dessa biblioteca. O `src/main.cpp` do projeto é um exemplo
completo e comentado do que está descrito aqui.

---

## 1. O que o modelo faz

| Propriedade | Valor |
|---|---|
| Tipo | **Regressão** (1 valor de saída, não classificação) |
| Entrada | **45 floats** = 5 amostras × 9 eixos |
| Saída | 1 float (`classification[0].value`) |
| Quantização | **int8** (zero point −121, escala 0.008002916) |
| Anomalia | **não tem** (o v4 tinha KMeans; foi removido no v5) |
| Motor | TFLite Micro via compilador EON, arena de 2732 bytes |
| Janela | `EI_CLASSIFIER_INTERVAL_MS` = 1000 ms, frequência 1 Hz |

Os 9 eixos, **nesta ordem** (é a ordem que o modelo espera):

```
0 temperatura_ambiente   3 hora_dec   6 ghit
1 doy                    4 kc         7 ioeff
2 ghi_cs                 5 angzen     8 kt
```

> ⚠️ **Não são dados de IMU.** Apesar do nome e do sensor estar declarado como `FUSION`, as entradas
> são variáveis de geometria solar e irradiância. Quem alimenta o modelo é você.

---

## 2. Estrutura da biblioteca

```
lib/knob-senses/
├── library.json          ← manifesto do PlatformIO (ver seção 3)
└── src/                  ← raiz de includes
    ├── edge-impulse-sdk/   o SDK de inferência
    ├── model-parameters/   metadados do SEU modelo (gerado pelo Studio)
    └── tflite-model/       o grafo compilado do SEU modelo (gerado pelo Studio)
```

Os três diretórios ficam juntos de propósito: os headers do SDK se incluem por caminho relativo à
raiz (`"model-parameters/model_variables.h"`), então `src/` precisa estar no include path.

**Para atualizar o modelo:** exporte de novo do Studio e descompacte por cima de
`lib/knob-senses/src/`. Nenhuma configuração precisa mudar.

---

## 3. Configuração do PlatformIO

Três coisas no `platformio.ini` são obrigatórias — as duas primeiras não são óbvias e o build falha
de formas confusas sem elas:

```ini
[env:esp32-s3-devkitc-1]
platform = espressif32
board = esp32-s3-devkitc-1
framework = arduino
monitor_speed = 115200

; 1. O modo LDF padrão (chain) casa includes pelo NOME da biblioteca, e nunca
;    encontra um include com caminho como "edge-impulse-sdk/classifier/...".
;    deep+ casa pelo arquivo que realmente existe dentro da biblioteca.
lib_ldf_mode = deep+

; 2. O SDK traz uma camada de portabilidade por plataforma (arduino, espressif,
;    mbed, zephyr, posix...) e escolhe uma por detecção automática. Duas podem
;    parecer ativas ao mesmo tempo e definir os mesmos símbolos ei_* duas vezes
;    -> erro de link. Fixamos a escolha aqui.
build_flags =
	-DEI_PORTING_ESPRESSIF=0
	; ^ as duas linhas acima são as obrigatórias para o EI. O resto do
	;   build_flags do projeto (PSRAM, USB CDC...) pode coexistir aqui.
```

> Ajuste `board` para o seu módulo. O que **não** pode faltar é `lib_ldf_mode` e
> `-DEI_PORTING_ESPRESSIF=0` — sem eles o build quebra de formas confusas (seção 9).

E no `lib/knob-senses/library.json`, o `"frameworks"` é obrigatório:

```json
{
  "frameworks": "arduino"
}
```

**Por quê:** o SDK traz `porting/mbed/` junto com `porting/arduino/`. Sem o manifesto, o PlatformIO
fareja o framework pela árvore de arquivos, conclui que a biblioteca é *mbed*, e a **descarta
silenciosamente** porque o projeto é `framework = arduino`. O sintoma é um `No such file or
directory` em um header que está bem ali na pasta.

---

## 4. A API — o fluxo mínimo

Toda a API pública sai de um único include:

```cpp
#include <edge-impulse-sdk/classifier/ei_run_classifier.h>
```

> ⚠️ **Inclua esse header em exatamente UM arquivo `.cpp` do projeto.**
>
> O SDK é header-only, mas com definições **não-`inline`** — a própria `run_classifier()`,
> `run_classifier_init()`, `init_impulse()` e até a variável `impulse_handle_1086855_1` moram dentro
> de headers. Se dois `.cpp` incluírem `ei_run_classifier.h`, o linker acusa `multiple definition`
> de dezenas de símbolos de uma vez.
>
> Se precisar chamar a inferência de vários arquivos, faça um wrapper: **um** `.cpp` inclui o header
> do EI e expõe funções suas; os outros incluem apenas o seu header.

O fluxo tem 4 passos: **montar o frame → criar o `signal_t` → chamar `run_classifier()` → ler o
resultado.**

```cpp
#include <Arduino.h>
#include <edge-impulse-sdk/classifier/ei_run_classifier.h>

void setup() {
    Serial.begin(115200);

    // 1. Monte o frame: 45 floats (5 amostras de 9 eixos) na ORDEM do modelo.
    float frame[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
    frame[0] = 27.3f;   // temperatura_ambiente
    frame[1] = 184.0f;  // doy
    frame[2] = 663.0f;  // ghi_cs
    frame[3] = 11.5f;   // hora_dec
    frame[4] = 0.74f;   // kc
    frame[5] = 44.1f;   // angzen
    frame[6] = 499.0f;  // ghit
    frame[7] = 1368.0f; // ioeff
    frame[8] = 0.51f;   // kt
    // ... preencha os índices 9..44 com as amostras 1 a 4 da janela
}

void loop() {
    float frame[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
    // preencha o frame com dados reais aqui

    // 2. Empacote o buffer em um signal_t.
    //    'frame' precisa continuar vivo durante a chamada. O EI normaliza
    //    (z = (x - mean) * scale) internamente — mande valores BRUTOS.
    signal_t signal;
    if (numpy::signal_from_buffer(frame, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE,
                                  &signal) != EIDSP_OK) {
        Serial.println("erro ao criar o signal");
        return;
    }

    // 3. Rode a inferência.
    ei_impulse_result_t result = { 0 };
    EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false /* debug */);
    if (err != EI_IMPULSE_OK) {
        Serial.print("erro na inferência: ");
        Serial.println((int)err);
        return;
    }

    // 4. Leia o resultado. É REGRESSÃO: o valor está em classification[0].value.
    Serial.print(result.classification[0].label);   // "value"
    Serial.print(" = ");
    Serial.println(result.classification[0].value, 4);

    Serial.print("dsp ");
    Serial.print(result.timing.dsp);
    Serial.print(" ms, inferência ");
    Serial.print(result.timing.classification);
    Serial.println(" ms");

    delay(EI_CLASSIFIER_INTERVAL_MS);
}
```

### Assinaturas

```cpp
// Cria um signal_t a partir de um buffer em RAM. Retorna EIDSP_OK em caso de sucesso.
static int numpy::signal_from_buffer(const float *data, size_t data_size, signal_t *signal);

// Inferência de uma janela completa. 'debug' imprime logs internos via ei_printf().
EI_IMPULSE_ERROR run_classifier(signal_t *signal, ei_impulse_result_t *result, bool debug = false);
```

### O resultado

```cpp
typedef struct {
    const char *label;   // rótulo da saída
    float       value;   // ← a predição (regressão)
} ei_impulse_result_classification_t;

// Dentro de ei_impulse_result_t:
result.classification[0].label;      // "value"
result.classification[0].value;      // a predição
result.timing.dsp;                   // ms no pré-processamento
result.timing.classification;        // ms na inferência
result.timing.anomaly;               // existe, mas sempre 0 no v5
```

> **`result.anomaly` continua existindo no struct mas é sempre 0** neste modelo — o bloco KMeans foi
> removido no deploy v5. Não use esse campo para tomar decisão.

---

## 5. O layout do frame (leia isto antes de alimentar o modelo)

O impulso declara `RAW_SAMPLE_COUNT = 5` e `RAW_SAMPLES_PER_FRAME = 9`, ou seja, uma janela de
**5 amostras × 9 eixos = 45 floats**, em ordem **sample-major**:

```
[s0.a0, s0.a1, ... s0.a8,  s1.a0, s1.a1, ... s1.a8,  ...  s4.a8]
 └───── amostra 0 ─────┘   └───── amostra 1 ─────┘
```

O bloco DSP é `extract_raw_features`, que **copia o buffer inteiro sem reordenar nada** (ele ignora
a tabela de eixos). Depois o SDK aplica o scaler padrão em `z = (x - mean) * scale`.

### ⚠ O modelo, como está treinado, não usa as 5 amostras

Esta é a parte que mais importa, e vale entender o mecanismo inteiro.

No `model-parameters/model_variables.h`, das 45 posições do scaler, só as **9 primeiras** têm
estatística real. As posições 9..44 estão com `mean = 0`, `scale = 1` e `var = 0` — variância
**exatamente** zero é a assinatura de colunas constantes no treino. Ou seja: o modelo foi treinado
com as 9 features na primeira amostra e **zeros atrás**.

A cadeia que isso destrava:

1. Nesses slots a normalização vira `z = (x − 0) × 1 = x` — **valor bruto passa direto, sem escala**.
2. O tensor de entrada é int8 com `scale = 0.0273695`, `zero_point = −20`
   (`tflite_learn_1086855_3_compiled.cpp:158-160`), logo a faixa representável é **[-2.96, +4.02]**.
3. A quantização **satura**, não dá a volta.

Então, se você mandar uma janela de 5 amostras **reais**, valores como `ioeff = 1367`, `ghi_cs = 663`
ou `doy = 184` caem nos slots 9..44 e **saturam em `+127`** — 36 dos 45 inputs fora da distribuição,
a cada inferência. No treino esses slots valiam `0` (→ `q = −20`, meio da faixa).

Na prática: a predição continua variando com os dados, mas com um **viés constante desconhecido**, e
quem a governa é a **amostra mais antiga** (slots 0..8) — o dado de 4 segundos atrás.

Duas leituras possíveis, e você precisa decidir qual é a sua:

- **Se a janela de 5 amostras é intencional**, seu dataset no Studio só populou a primeira amostra.
  Vale corrigir o dataset (e o scaler) e re-treinar.
- **Se foi uma configuração que passou batido**, o modelo efetivamente usa 9 entradas e o resto é
  inofensivo desde que você mande zeros.

**Como testar as duas:** o `src/main.cpp` tem a constante `kZeroTail`:

```cpp
static constexpr bool kZeroTail = false;  // true = zeros atrás, como treinado
```

`false` monta a janela real de 5 amostras; `true` zera os 36 slots do fim. Rode os dois e compare as
predições — se só o `true` der valores coerentes, o achado está confirmado.

Confirme no Studio antes de confiar cegamente nos resultados.

---

## 6. Macros de introspecção

Não hardcode nada disso — o modelo é re-gerado e os valores mudam. Use as macros de
`model-parameters/model_metadata.h`:

| Macro | Valor no v5 | Para que serve |
|---|---|---|
| `EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE` | 45 | Tamanho do buffer de entrada |
| `EI_CLASSIFIER_RAW_SAMPLE_COUNT` | 5 | Amostras na janela |
| `EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME` | 9 | Eixos por amostra |
| `EI_CLASSIFIER_LABEL_COUNT` | 1 | Quantas saídas |
| `EI_CLASSIFIER_INTERVAL_MS` | 1000 | Intervalo entre janelas |
| `EI_CLASSIFIER_FUSION_AXES_STRING` | `"temperatura_ambiente + doy + ..."` | A ordem dos eixos |
| `EI_CLASSIFIER_PROJECT_NAME` | `"knob-senses"` | Nome do projeto |
| `EI_CLASSIFIER_PROJECT_DEPLOY_VERSION` | 5 | Versão do deploy |
| `EI_CLASSIFIER_HAS_ANOMALY` | `EI_ANOMALY_TYPE_UNKNOWN` | Se há bloco de anomalia |

Um uso defensivo comum é travar o contrato em tempo de compilação:

```cpp
static_assert(EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE ==
              EI_CLASSIFIER_RAW_SAMPLE_COUNT * EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME,
              "janela do impulso inconsistente");
```

---

## 7. Inferência contínua — existe, mas **não use neste impulso**

`run_classifier()` é **bloqueante** e processa uma janela inteira por chamada — é o caminho usado
acima e no `src/main.cpp`. Para este modelo, é o caminho certo: o `src/main.cpp` monta a janela de
45 floats num buffer circular próprio e entrega ela pronta a cada segundo.

O modo contínuo (`run_classifier_continuous()`) existe e compila, mas **não se aplica aqui**:

```
EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW = 4
EI_CLASSIFIER_SLICE_SIZE = (5 / 4) = 1        ← divisão inteira
```

O modo contínuo pressupõe que `RAW_SAMPLE_COUNT` seja **múltiplo** de `SLICES_PER_MODEL_WINDOW`.
Com 5 amostras divididas em 4 fatias o resultado é 1 amostra por fatia — a janela de 5 nunca é
preenchida de forma limpa. Não é um recurso que você deva usar sem antes re-treinar o impulso com
uma janela divisível.

Se um dia fizer sentido (após re-treinar), a API é esta — as assinaturas abaixo foram verificadas
por compilação:

```cpp
#include <edge-impulse-sdk/classifier/ei_run_classifier.h>

void setup() {
    run_classifier_init();   // zera o estado interno; obrigatório antes do contínuo
}

void loop() {
    float slice[EI_CLASSIFIER_SLICE_SIZE];
    // ... preencha a fatia ...

    signal_t signal;
    numpy::signal_from_buffer(slice, EI_CLASSIFIER_SLICE_SIZE, &signal);

    ei_impulse_result_t result = { 0 };
    run_classifier_continuous(&signal, &result, false);
    // Nas primeiras (SLICES_PER_MODEL_WINDOW - 1) chamadas a janela ainda está
    // enchendo e a saída não é válida.
}
```

Chame `run_classifier_init()` de novo se o fluxo de dados for interrompido. Existe também a
sobrecarga `run_classifier_init(&ei_default_impulse)` e `run_classifier_deinit()`, para quando você
mantém múltiplos impulsos.

---

## 8. Tratamento de erros

`run_classifier()` devolve um `EI_IMPULSE_ERROR`. Os que você provavelmente vai encontrar:

| Código | Significado |
|---|---|
| `EI_IMPULSE_OK` (0) | Sucesso |
| `-1` / `-24` | Formato/tamanho dos dados não bate com a entrada (`SHAPES_DONT_MATCH` / `INVALID_SIZE`) |
| `-5` | Erro no bloco de processamento (DSP) |
| `-3` / `-22` | Erro na inferência TFLite |
| `-6` / `-8` / `-11` | Falha/dificuldade de alocação de memória |
| `-20` | Modelo descontinuado — re-exporte do Studio |
| `-30` | Erro na normalização de dados |

O código `-11` (`OUT_OF_MEMORY`) e o `-6` (`TFLITE_ARENA_ALLOC_FAILED`) costumam indicar heap
fragmentado, não falta de RAM — a arena deste modelo tem só 2732 bytes.

---

## 9. Pegadinhas já resolvidas (e que voltam se você mexer)

| Sintoma | Causa | Solução |
|---|---|---|
| `No such file or directory` num header que existe | `lib_ldf_mode` no padrão `chain` | `lib_ldf_mode = deep+` |
| A biblioteca some do build, sem aviso | `library.json` sem `"frameworks": "arduino"` | Declare o framework |
| `multiple definition of 'ei_printf'` e afins | Duas camadas de portabilidade ativas | `-DEI_PORTING_ESPRESSIF=0` |
| `multiple definition` de dezenas de símbolos (`run_classifier`, `init_impulse`...) | `ei_run_classifier.h` incluído em mais de um `.cpp` | Inclua em um único `.cpp` e exponha um wrapper |
| `multiple definition of 'tflite::...'` | Outra lib trazendo um TensorFlow Lite Micro próprio | Não misture (ex.: `Harvard_TinyMLx`) |
| Saída com números sem sentido | Frame preenchido fora de ordem, ou valores pré-normalizados | Seção 5 — mande valores **brutos**, na ordem certa |
| Predição com viés, mas ainda reagindo aos dados | Saturação dos 36 slots do fim (`kZeroTail`) | Seção 5 — o modelo não foi treinado para janela real |

---

## 10. Como usar no hardware

O `src/main.cpp` é um **app de streaming**: recebe uma amostra por segundo, guarda as 5 últimas num
buffer circular e roda uma inferência a cada amostra recebida.

```bash
pio run -t upload
pio device monitor
```

Saída esperada, do boot até o regime:

```
knob-senses v5 — janela 5x9 = 45 floats, 1 inferencia por segundo
ordem dos eixos: 0=temperatura_ambiente, 1=doy, 2=ghi_cs, 3=hora_dec, 4=kc,
                 5=angzen, 6=ghit, 7=ioeff, 8=kt
ZERO_TAIL = false (janela real de 5 amostras)
fonte: DEMO (rampa sintetica) - implemente readSample() para dados reais

janela 1/5 - enchendo          <- t=1s
janela 2/5 - enchendo          <- t=2s
janela 3/5 - enchendo          <- t=3s
janela 4/5 - enchendo          <- t=4s
  t-4s | 0.0  1.0  2.0 ...     <- t=5s: primeira inferência
  t-3s | 10.0 11.0 12.0 ...
  t-2s | 20.0 21.0 22.0 ...
  t-1s | 30.0 31.0 32.0 ...
  t   | 40.0 41.0 42.0 ...
  ordem: OK
  pred = 23.4567   dsp 0 ms  inf 1 ms  heap_livre 412340  overruns 0
```

**O gancho de aquisição é `readSample()`,** no fim do arquivo. Enquanto ele não estiver implementado,
`SAMPLE_SOURCE` aponta para uma demo que gera uma rampa sintética — o suficiente para validar todo o
pipeline sem sensor. Troque para `SAMPLE_SOURCE_USER` quando implementar a leitura real.

A demo usa uma rampa **de propósito**: ela codifica a idade de cada amostra (`valor = seq*10 + eixo`),
então o frame imprime com passo 1 dentro da linha e 10 entre linhas. Isso torna qualquer erro de
ordem visível na hora — e o veredito `ordem: OK` / `ordem: BAD` é calculado sobre o frame. Uma
senoide não serviria para isso (é invariante sob rotação e reversão, a menos de fase).

**Assinaturas de erro de ordem**, todas visíveis nas linhas impressas:

| O que aparece | Bug |
|---|---|
| bases das linhas **decrescendo** | ordem invertida |
| um salto negativo grande caminhando entre ticks | usou `head` como mais recente |
| as 5 linhas idênticas | `head` não avança |
| passo **5** dentro da linha, e não 1 | eixos intercalados (`j*kSamples + k` em vez de `k*kAxes + j`) |

**Estabilidade:** o campo `heap_livre` é o maior bloco contíguo do heap. Deixe rodando uns 30 min —
se ficar plano, não há fragmentação nem vazamento; se cair continuamente, há. O campo `overruns`
conta ticks em que o loop atrasou um período inteiro e deve ficar em 0.

> **Estado atual:** o firmware **compila e linka** (RAM 6.5%, Flash 4.6%) e a lógica do buffer
> circular foi verificada em teste nativo (1020 ticks, ~200 voltas completas). Mas a **inferência não
> foi validada em hardware** — não há placa nesta sessão. Os passos acima são para você rodar.

### Voltar ao harness interativo

O harness antigo (comandos `mean` / `sweep` / `<45 números>`, para comparar com o CSV de teste) foi
**substituído** por este app — o projeto não é um repositório git, então não há como recuperá-lo
sozinho. Se quiser os dois, o caminho é separar em arquivos: o harness vira um `.cpp` próprio atrás
de um header-wrapper, porque `ei_run_classifier.h` só pode ser incluído em **um** arquivo por projeto
(seção 4).

Para validar contra o dataset sem o harness, use `readSample()`: cole uma linha do CSV de teste nas
9 features e deixe rodar — mas lembre do `kZeroTail` (seção 5) antes de confiar no número.
