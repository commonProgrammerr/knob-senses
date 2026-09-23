/*
 * knob-senses — inferência de regressão em streaming
 *
 *   projeto 1086855 / "Impulse #1" / deploy v5
 *   grafo EON int8: janela de 5 amostras x 9 features -> 1 valor de regressão
 *   (o v5 removeu o bloco KMeans de anomalia que o v4 tinha)
 *
 * Comportamento: uma amostra por segundo (9 features) entra num buffer
 * circular de 5 amostras — uma janela completa do impulso — e uma inferência
 * roda a cada amostra recebida. Nos 4 primeiros segundos a janela ainda está
 * enchendo e nada é inferido; a primeira predição sai em t = 5 s.
 *
 * >>>>> O GANCHO DE AQUISIÇÃO É readSample(), MAIS ABAIXO <<<<<
 * Enquanto ele não estiver implementado, SAMPLE_SOURCE aponta para uma fonte
 * sintética que permite testar todo o pipeline sem sensor.
 *
 * ---------------------------------------------------------------------------
 * Ordem do frame
 *
 * O impulso declara EI_CLASSIFIER_RAW_SAMPLE_COUNT 5 e
 * EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME 9, então o sinal é de 5 * 9 = 45 floats.
 * O bloco DSP é extract_raw_features, que copia o buffer inteiro sem reordenar
 * (ignora a tabela de eixos), e o scaler padrão aplica z = (x - mean) * scale
 * nos 45 slots.
 *
 * A ordem aqui é CRONOLÓGICA, que é a convenção do Edge Impulse:
 *
 *   slots  0.. 8 = t-4s  (mais antiga)
 *   slots  9..17 = t-3s
 *   slots 18..26 = t-2s
 *   slots 27..35 = t-1s
 *   slots 36..44 = t     (mais recente)
 *
 * ---------------------------------------------------------------------------
 * ATENÇÃO: este modelo, como está treinado, não parece usar as 5 amostras
 *
 * Os slots 9..44 do scaler têm mean 0, var 0 e scale 1
 * (lib/knob-senses/src/model-parameters/model_variables.h:61-63). Variância
 * exatamente zero é a assinatura de colunas constantes no treino: o modelo foi
 * treinado com as 9 features na primeira amostra e zeros atrás.
 *
 * Isso importa porque, nesses slots, a normalização vira (x - 0) * 1 = x —
 * valor BRUTO passando direto, sem escala. E o tensor de entrada é int8 com
 * scale 0.0273695 / zero_point -20 (tflite_learn_1086855_3_compiled.cpp:158-160),
 * ou seja faixa representável [-2.96, +4.02]. Como a quantização satura em vez
 * de dar a volta, valores brutos como ioeff = 1367 ou ghi_cs = 663 caem em
 * +127 no tensor.
 *
 * Consequência prática: com uma janela de 5 amostras REAIS, 36 dos 45 inputs
 * saturam a cada inferência, enquanto no treino valiam 0 (-> q = -20). A
 * predição continua variando com os dados, mas com um viés constante
 * desconhecido, e quem a governa é a amostra mais antiga (slots 0..8).
 *
 * O switch ZERO_TAIL abaixo permite comparar os dois enquadramentos no mesmo
 * binário. Se só o modo `true` der valores coerentes, o achado está confirmado
 * e o modelo precisa ser re-treinado com janelas de verdade antes de usar
 * janela real. Veja a seção 5 do README.md.
 */

#include <Arduino.h>
#include <string.h>
#include <WiFi.h>
#include <WiFiClient.h>

#include <edge-impulse-sdk/classifier/ei_run_classifier.h>

// #define SERVER_IP "10.82.109.79"
// #define SERVER_IP "10.0.18.81"
#define SERVER_IP "192.168.1.101"
#define SERVER_PORT 3333
#define WIFI_SSID "Taluana"
#define WIFI_PASSWORD "DLNT1806"

/* ------------------------------------------------------------------ layout */

/* Tudo derivado do modelo — nunca hardcode estes números. */
static constexpr size_t kSamples = EI_CLASSIFIER_RAW_SAMPLE_COUNT;       // 5
static constexpr size_t kAxes = EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME;     // 9
static constexpr size_t kFrameSize = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; // 45
static constexpr uint32_t kPeriodMs = 50;                                // EI_CLASSIFIER_INTERVAL_MS;         // 1000

static_assert(kFrameSize == kSamples * kAxes,
              "janela do impulso nao bate com RAW_SAMPLE_COUNT * RAW_SAMPLES_PER_FRAME");
static_assert(EI_CLASSIFIER_LABEL_COUNT == 1, "esperada 1 saida de regressao");

/* Armadilha, não bug: dispara se um dia você re-treinar com janela divisível
 * por SLICES_PER_MODEL_WINDOW, que é quando run_classifier_continuous() passa
 * a ser utilizável (README seção 7). Hoje: 5 % 4 = 1. */
// static_assert(EI_CLASSIFIER_RAW_SAMPLE_COUNT % EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW != 0,
//               "janela agora e divisivel: run_classifier_continuous() passou a ser usavel");

/*
 * false = janela real de 5 amostras (o padrão, e o alvo depois de re-treinar)
 * true  = zeros nos 36 slots do fim, que é como este modelo foi treinado
 *
 * Só muda a montagem do frame; o buffer circular é idêntico nos dois casos.
 */
static constexpr bool kZeroTail = false;

/* --------------------------------------------------- fonte de amostras (gancho) */

#define SAMPLE_SOURCE_DEMO 0
#define SAMPLE_SOURCE_USER 1

/*
 * Qual fonte usar. Troque para SAMPLE_SOURCE_USER quando implementar a leitura
 * real — ou defina por build_flags (ex: -DSAMPLE_SOURCE=SAMPLE_SOURCE_USER).
 */
#define SAMPLE_SOURCE SAMPLE_SOURCE_USER

/*
 * ===========================================================================
 * O GANCHO
 *
 * Adquira UMA amostra das 9 features, na ordem abaixo, e retorne true.
 * Retorne false se neste tick ainda não há dado disponível (a janela avança
 * mesmo assim, de propósito — veja o comentário no loop).
 *
 *   0 temperatura_ambiente   4 kc       8 kt
 *   1 doy                    5 angzen
 *   2 ghi_cs                 6 ghit
 *   3 hora_dec               7 ioeff
 *
 * Mande valores BRUTOS, nas unidades originais. O impulso aplica o scaler
 * padrão internamente — não pré-normalize nada aqui.
 * ===========================================================================
 */
static bool readSample(float out[kAxes], WiFiClient &client);

/* --------------------------------------------------------------- estado */

static float g_ring[kSamples][kAxes]; // buffer circular
static size_t g_head = 0;             // índice do PRÓXIMO slot a escrever
static size_t g_filled = 0;           // 0..kSamples, saturando

static float g_frame[kFrameSize]; // janela linearizada, entregue ao SDK
static ei_impulse_result_t g_result;

static uint32_t g_nextDueMs = 0;
static uint32_t g_sampleSeq = 0; // amostras capturadas; usado pela demo
static uint32_t g_overruns = 0;  // ticks em que atrasamos um período inteiro
static uint32_t g_dropouts = 0;  // ticks sem amostra disponível
static bool g_inferenceFailed = false;
WiFiClient g_client; // Cria um cliente WiFi para enviar resultados

/* LED opcional: a placa pode não declarar LED_BUILTIN. Pisca a cada inferência. */
// #define LED_PIN 2

/* ------------------------------------------------------------ buffer circular */

static void ringPush(const float s[kAxes])
{
  memcpy(g_ring[g_head], s, kAxes * sizeof(float));
  g_head = (g_head + 1) % kSamples; // 5 NÃO é potência de 2: % e não &
  if (g_filled < kSamples)
  {
    g_filled++; // satura em kSamples — NÃO usar %= aqui
  }
}

/*
 * Lineariza o anel em ordem cronológica. Precondição: g_filled == kSamples.
 *
 * Com o anel cheio, g_head é o slot MAIS ANTIGO — é o próximo a ser
 * sobrescrito, logo o mais velho. O "+ kSamples" antes da subtração é o que
 * evita underflow quando head < filled.
 */
static void ringLinearize(float *out)
{
  const size_t oldest = (g_head + kSamples - g_filled) % kSamples;
  for (size_t k = 0; k < kSamples; k++)
  {
    memcpy(out + k * kAxes, g_ring[(oldest + k) % kSamples], kAxes * sizeof(float));
  }

  if (kZeroTail)
  {
    memset(out + kAxes, 0, (kFrameSize - kAxes) * sizeof(float));
  }
}

/* ---------------------------------------------------------------- inferência */

static bool runInference(const float *frame, ei_impulse_result_t *result)
{
  signal_t signal;
  if (numpy::signal_from_buffer(frame, kFrameSize, &signal) != EIDSP_OK)
  {
    Serial.println("erro: nao foi possivel montar o signal");
    return false;
  }

  /* process_impulse() ja zera o result; nao precisa memset aqui. */
  return run_classifier(&signal, result, false) == EI_IMPULSE_OK;
}

/* --------------------------------------------------------------------- debug */

static void printFrame(const float *frame)
{
  for (size_t k = 0; k < kSamples; k++)
  {
    Serial.print("  t-");
    Serial.print((int)(kSamples - 1 - k));
    Serial.print("s |");
    for (size_t j = 0; j < kAxes; j++)
    {
      Serial.print(' ');
      Serial.print(frame[k * kAxes + j], 1);
    }
    Serial.println();
  }
}

static void printResult(const ei_impulse_result_t &result)
{
  Serial.print("  pred = ");
  Serial.print(result.classification[0].value, 4);
  Serial.print("   dsp ");
  Serial.print(result.timing.dsp);
  Serial.print(" ms  inf ");
  Serial.print(result.timing.classification);
  Serial.print(" ms");

#if defined(ARDUINO_ARCH_ESP32)
  /* Se este número cair ao longo de horas há fragmentação de heap; plano é ok. */
  Serial.print("  heap_livre ");
  Serial.print((uint32_t)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
#endif

  Serial.print("  overruns ");
  Serial.println(g_overruns);
}

static void send_results(const ei_impulse_result_t &result, WiFiClient &client)
{
  if (!client.connected())
  {
    Serial.println("Falha na conexão com o servidor");
    return;
  }
  size_t total_request_size = sizeof(float) + 1U;
  const uint8_t *send_buf = (const uint8_t *)&(result.classification[0].value);

  size_t bytesWritten = (client.write((const uint8_t *)"\x02", 1U) +
                         client.write(send_buf, sizeof(float))); // send body

  if (bytesWritten != sizeof(float) + 1U)
  {
    Serial.println("Falha ao enviar resultados para o servidor");
    Serial.printf("Bytes enviados/esperados: %U/%U", bytesWritten, total_request_size);
  }
}

/* ---------------------------------------------------------------------- setup */

static void ledSetup()
{
#ifdef LED_PIN
  pinMode(LED_PIN, OUTPUT);
#endif
}

static void ledWrite(bool on)
{
#ifdef LED_PIN
  digitalWrite(LED_PIN, on ? HIGH : LOW);
#else
  (void)on;
#endif
}

static void printBanner()
{
  Serial.println();
  Serial.print("knob-senses v");
  Serial.print(EI_CLASSIFIER_PROJECT_DEPLOY_VERSION);
  Serial.print(" — janela ");
  Serial.print((int)kSamples);
  Serial.print("x");
  Serial.print((int)kAxes);
  Serial.print(" = ");
  Serial.print((int)kFrameSize);
  Serial.println(" floats, 1 inferencia por segundo");
  Serial.print("ordem dos eixos: ");
  static const char *kNames[kAxes] = {
      "temperatura_ambiente", "ghit", "jacques"};
  for (size_t i = 0; i < kAxes; i++)
  {
    if (i)
    {
      Serial.print(", ");
    }
    Serial.print((int)i);
    Serial.print("=");
    Serial.print(kNames[i]);
  }
  Serial.println();
  Serial.print("ZERO_TAIL = ");
  Serial.println(kZeroTail ? "true (zeros atras, como treinado)"
                           : "false (janela real de 5 amostras)");
}

void setup()
{
  ledSetup();

  Serial.begin(115200);
  /* Sem while (!Serial): este e um app standalone e nao pode bloquear
   * esperando um monitor que talvez nunca conecte. O banner sai a cada
   * tick, entao nada se perde se voce conectar depois. */
  printBanner();
  /*
   * SSID:	POCO X7 Pro
   * Protocolo:	Wi-Fi 4 (802.11n)
   * Tipo de segurança:	WPA2-Personal
   * Banda de rede (canal):	2,4 GHz (1)
   * Velocidade do link agregado (Recepção/Transmissão):	13/144 (Mbps)
   */
  // WiFi.begin("POCO X7 Pro", "rszrv5ip96vhag3");
  // WiFi.begin("Taluana", "DLNT1806");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Conectando ao WiFi");
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }
  Serial.println("WiFi conectado");
}
/* ----------------------------------------------------------------------- loop */

void loop()
{
  const uint32_t now = millis();
  /* Subtração SEM SINAL: é o único jeito seguro com millis(), que dá a volta
   * em ~49,7 dias. Comparar millis() >= deadline quebraria no wrap. */
  if ((uint32_t)(now - g_nextDueMs) < kPeriodMs)
  {
    return;
  }

  /* Passo fixo (+=) e não = now: mantém a fase de 1 Hz em vez de acumular o
   * tempo de execução da inferência e da serial. Aqui isso importa de
   * verdade — hora_dec e angzen são features de hora do dia, então deriva de
   * relógio é erro silencioso de entrada. */
  g_nextDueMs += kPeriodMs;
  if ((uint32_t)(millis() - g_nextDueMs) >= kPeriodMs)
  {
    g_overruns++; /* atrasamos um período inteiro: nao tenta recuperar em rajada */
    g_nextDueMs = millis();
  }

  /**
   * Connect to sever (simulates data aquisition)
   */
  g_client.connect(SERVER_IP, SERVER_PORT);

  float sample[kAxes];
  if (!readSample(sample, g_client))
  {
    /* O deadline já avançou de propósito: a cadência de 1 Hz é o contrato
     * do modelo, então a janela anda mesmo sem dado e o dropout é contado.
     * (A alternativa — não avançar — deslocaria toda janela seguinte.) */
    g_dropouts++;
    Serial.print("sem amostra (dropout ");
    Serial.print(g_dropouts);
    Serial.println(")");
    return;
  }

  g_sampleSeq++;
  ringPush(sample);

  if (g_filled < kSamples)
  {
    Serial.print("janela ");
    Serial.print((int)g_filled);
    Serial.print("/");
    Serial.print((int)kSamples);
    Serial.println(" - enchendo");
    g_client.connect(SERVER_IP, SERVER_PORT);
    g_client.write((const uint8_t *)"\x00", 1); // skip send;
    return;
  }

  if (g_inferenceFailed)
  {
    return; /* veja o comentario do latch abaixo */
  }

  ledWrite(true);

  ringLinearize(g_frame);

  /* Imprime antes de inferir: a evidência da ordem sobrevive a um erro. */
  // printFrame(g_frame);

  if (!runInference(g_frame, &g_result))
  {
    /* FATAL, e não log-and-continue. Os caminhos de erro de
     * inferencing_engines/tflite_eon.h (~linha 243) retornam sem chamar
     * model_reset(ei_aligned_free) nem ei_free(outputs), ou seja vazam a
     * arena de 2732 B por chamada. A 1 Hz isso esgota o heap em minutos. */
    g_inferenceFailed = true;
    Serial.println("ERRO FATAL na inferencia - parando de inferir");
    Serial.println("(insistir vazaria a arena do TFLite a cada segundo)");
    ledWrite(false);
    return;
  }
  g_client.connect(SERVER_IP, SERVER_PORT);

  printResult(g_result);
  send_results(g_result, g_client);
  ledWrite(false);
  g_client.stop();
}

static bool readSample(float out[kAxes], WiFiClient &client)
{
  if (!client.connected() || !client.write((const uint8_t *)"\x01", 1U))
  {
    Serial.println("Falha na conexão com o servidor");
    return false;
  }
  size_t bytesExpected = kAxes * sizeof(float);
  size_t bytesRead = client.readBytes((char *)out, bytesExpected);
  if (bytesRead != kAxes * sizeof(float))
  {
    Serial.println("Falha na leitura completa do servidor");
    Serial.printf("Bytes lidos/esperados: %u/%ub\n", bytesRead, bytesExpected);
    return false;
  }
  return client.write((const uint8_t *)"\x01", 1) > 0; // envia ACK;
}
