#include <Arduino.h>
#include <string.h>
#include <WiFi.h>
#include <WiFiClient.h>

/* O modelo ativo vem do ENV do PlatformIO (platformio.ini): um env por modelo, com as
 * flags especificas isoladas (ex.: o Edge Impulse so' no mlp). Nao troque o include a
 * mao - escolha o env:  pio run -e rf | -e mlp | -e knn  */
#if defined(MODEL_RF)
#include "rf_model.h"

#define runInference(frame, result) rf_run_inference(frame, result)
#define printResult(result) rf_print_result(result)
#define printBanner() rf_print_banner()
typedef rf_result_t result_t;

#elif defined(MODEL_MLP)
#include "mlp_model.h"

#define runInference(frame, result) mlp_run_inference(frame, result)
#define printResult(result) mlp_print_result(result)
#define printBanner() mlp_print_banner()
typedef ei_impulse_result_t result_t;

#elif defined(MODEL_KNN)
#include "knn_model.h"

#define runInference(frame, result) knn_run_inference(frame, result)
#define printResult(result) knn_print_result(result)
#define printBanner() knn_print_banner()
typedef knn_result_t result_t;

#else
#error "Nenhum modelo: use um env do platformio.ini (-e rf | -e mlp | -e knn)"
#endif

#define SERVER_IP "192.168.1.102"
#define SERVER_PORT 3333
#define WIFI_SSID "Taluana"
#define WIFI_PASSWORD "DLNT1806"

/* ---------------------------------------------- protocolo v2 (README secao 4)
 *
 * A sessao inteira vive em UMA conexao TCP: o cliente pede a amostra `seq`, infere e
 * devolve a predicao da MESMA seq; o servidor confirma com um ACK depois de gravar a
 * linha. O numero de sequencia volta ecoado em toda resposta - sem ele, uma perda
 * desalinhava o log em silencio (real e pred trocados por linha) e ninguem via.
 *
 * A seq so' avanca com o ACK. Qualquer falha (connect, escrita parcial, leitura
 * incompleta, seq ou tipo diferentes) => stop() e a MESMA amostra volta no proximo
 * tick: nunca se pula amostra. O mesmo byte de tipo vale nos dois sentidos - 0x01 e'
 * pedido no sentido ESP->servidor e amostra no contrario; 0x03 e' predicao e ACK.
 */
#define MSG_REQUEST 0x01    /* ESP -> servidor: 0x01 + seq u32 (5 bytes)             */
#define MSG_SAMPLE 0x01     /* servidor -> ESP: 0x01 + seq u32 + kAxes x float32     */
#define MSG_END 0x02        /* servidor -> ESP: 0x02 + seq u32 (nao ha' seq < len)   */
#define MSG_PREDICTION 0x03 /* ESP -> servidor: 0x03 + seq u32 + pred float32 (9 B)  */
#define MSG_ACK 0x03        /* servidor -> ESP: 0x03 + seq u32 (linha `seq` logada)  */

/* WiFiClient::setTimeout() recebe SEGUNDOS (o core multiplica por 1000) e vale para o
 * connect E para o readBytes (Stream::_timeout, POR byte; o SO_RCVTIMEO que ele tambem
 * seta nao e' consultado pelo leitor). 2 s: o servidor tem 5 s, entao quem desiste
 * primeiro e' sempre o firmware - e' ele que reconecta e retenta. */
#define SERVER_TIMEOUT_S 2U

/* ------------------------------------------------------------------ estado */

static uint32_t g_nextDueMs = 0;
static uint32_t g_seq = 0;      /* proxima amostra a pedir: so' avanca com o ACK */
static bool g_done = false;     /* o servidor avisou o fim do dataset */
static bool g_serverOk = true;  /* false apos uma falha; true de proposito, para o
                                 * primeiro "servidor fora do ar" imprimir a transicao */
static uint32_t g_overruns = 0; /* ticks em que atrasamos um periodo inteiro */
static uint32_t g_dropouts = 0; /* trocas falhadas, que serao RETENTADAS (a seq nao avanca) */
static bool g_inferenceFailed = false;
result_t g_result;
WiFiClient g_client;

/* O veredito de uma troca: e' o que o loop usa para decidir a proxima acao. */
enum Exchange
{
  EXCHANGE_OK,            /* amostra recebida, predicao logada e ACK conferido */
  EXCHANGE_END,           /* o servidor nao tem esta seq: acabou o dataset */
  EXCHANGE_ERR_WRITE,     /* write() parcial ou zero */
  EXCHANGE_ERR_READ,      /* readBytes() incompleto (timeout / conexao caida) */
  EXCHANGE_ERR_SEQ,       /* o seq que voltou nao e' o que foi pedido */
  EXCHANGE_ERR_TYPE,      /* tipo de mensagem inesperado para este momento */
  EXCHANGE_ERR_ACK,       /* a resposta a predicao nao foi o ACK da seq */
  EXCHANGE_ERR_INFERENCE, /* runInference() falhou: fatal, para de inferir */
};

static bool ensure_server(WiFiClient &client);
static void record_failure(const char *reason);
static void record_success(void);
static const char *exchange_reason(Exchange outcome);
static Exchange exchange_sample(void);

/* ------------------------------------------------------------------- setup */

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

void setup()
{
  ledSetup();

  Serial.begin(115200);

  printBanner();

  g_client.setTimeout(SERVER_TIMEOUT_S); /* segundos! vale para o connect e para cada byte lido */

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Conectando ao WiFi");
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }
  Serial.println("WiFi conectado");
}

/* -------------------------------------------------------------------- loop */

void loop()
{
  const uint32_t now = millis();
  if ((uint32_t)(now - g_nextDueMs) < kPeriodMs)
  {
    return;
  }

  g_nextDueMs += kPeriodMs;
  if ((uint32_t)(millis() - g_nextDueMs) >= kPeriodMs)
  {
    g_overruns++; /* atrasamos um periodo inteiro: nao tenta recuperar em rajada */
    g_nextDueMs = millis();
  }

  if (g_done)
  {
    return; /* dataset inteiro: parou de pedir (a conexao ja' foi fechada no MSG_END) */
  }
  if (g_inferenceFailed)
  {
    return;
  }
  if (!ensure_server(g_client))
  {
    record_failure("sem conexao com o servidor");
    return;
  }

  ledWrite(true);
  const Exchange outcome = exchange_sample();
  ledWrite(false);

  if (outcome == EXCHANGE_END)
  {
    Serial.print("fim do dataset (o servidor nao tem a seq ");
    Serial.print(g_seq);
    Serial.println(") - nada mais a pedir");
    g_client.stop();
    g_done = true;
    return;
  }
  if (outcome == EXCHANGE_ERR_INFERENCE)
  {
    g_inferenceFailed = true;
    Serial.println("ERRO FATAL na inferencia - parando de inferir");
    Serial.println("(insistir vazaria a arena do TFLite a cada segundo)");
    g_client.stop(); /* a amostra ficou sem predicao: a seq NAO avanca */
    return;
  }
  if (outcome != EXCHANGE_OK)
  {
    record_failure(exchange_reason(outcome));
    return;
  }

  record_success(); /* so' AQUI: o servidor confirmou o log desta seq */
  printResult(g_result);
  g_seq++;
}

/* -------------------------------------------------- funcoes do protocolo */

static void put_u32(uint8_t *p, uint32_t v)
{
  /* LE explicito: o contrato do protocolo nao depende do alvo ser little-endian */
  p[0] = (uint8_t)(v & 0xFFU);
  p[1] = (uint8_t)((v >> 8) & 0xFFU);
  p[2] = (uint8_t)((v >> 16) & 0xFFU);
  p[3] = (uint8_t)((v >> 24) & 0xFFU);
}

static uint32_t get_u32(const uint8_t *p)
{
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

/* write() devolve o que conseguiu enviar - pode ser PARCIAL - e devolve 0 com a conexao
 * caida. Mensagem so' sai inteira com este laco; parcial e' falha, e falha e' stop() +
 * a MESMA amostra no proximo tick. */
static bool write_all(WiFiClient &client, const uint8_t *buf, size_t len)
{
  size_t sent = 0;
  while (sent < len)
  {
    const size_t n = client.write(buf + sent, len - sent);
    if (n == 0U)
    {
      return false;
    }
    sent += n;
  }
  return true;
}

/* readBytes() para no timeout e devolve o que veio (o timeout e' POR byte). Mensagem de
 * tamanho fixo nao tem leitura parcial aproveitavel: ou veio tudo, ou e' falha. */
static bool read_all(WiFiClient &client, uint8_t *buf, size_t len)
{
  return client.readBytes((char *)buf, len) == len;
}

/* Mantem UMA conexao viva pela sessao inteira - e' o unico lugar que conecta. O
 * connected() so' olha o socket local: servidor morto sem RST continua "conectado", e
 * por isso quem decide a falha e' a troca (timeout de leitura), nao este pre-filtro. */
static bool ensure_server(WiFiClient &client)
{
  if (client.connected())
  {
    return true;
  }
  client.stop();
  if (!client.connect(SERVER_IP, SERVER_PORT))
  {
    return false;
  }
  client.setNoDelay(true); /* mensagens de 5..25 B: o Nagle seguraria o ACK */
  return true;
}

/* Falha = a MESMA amostra volta no proximo tick: conta, fecha a conexao (a tentativa
 * recomeca do zero) e imprime SO' a transicao - a 40 Hz, um print por tick vira parede
 * de log e esconde o que importa. */
static void record_failure(const char *reason)
{
  g_dropouts++;
  g_client.stop();
  if (g_serverOk)
  {
    Serial.print("troca falhou (");
    Serial.print(reason);
    Serial.print(") - retenta a MESMA amostra; a seq nao avanca. dropouts=");
    Serial.println(g_dropouts);
    g_serverOk = false;
  }
}

/* O ACK chegou: o log do servidor tem esta seq. */
static void record_success(void)
{
  if (!g_serverOk)
  {
    Serial.print("servidor respondeu de novo (trocas falhadas ate agora: ");
    Serial.print(g_dropouts);
    Serial.println(")");
    g_serverOk = true;
  }
}

static const char *exchange_reason(Exchange outcome)
{
  switch (outcome)
  {
  case EXCHANGE_ERR_WRITE:
    return "escrita incompleta";
  case EXCHANGE_ERR_READ:
    return "leitura incompleta (timeout)";
  case EXCHANGE_ERR_SEQ:
    return "seq ecoado diferente do pedido";
  case EXCHANGE_ERR_TYPE:
    return "tipo de resposta inesperado";
  case EXCHANGE_ERR_ACK:
    return "resposta a predicao nao foi o ACK";
  case EXCHANGE_ERR_INFERENCE:
    return "inferencia";
  default:
    return "?";
  }
}

/* Pede a amostra `seq` e le a resposta em DUAS fases: os 5 bytes de cabecalho (tipo +
 * seq) primeiro, e so' depois, se for amostra, os kAxes floats. Ler tudo de uma vez
 * travaria na resposta de fim (5 bytes): o servidor nao manda mais nada e so' haveria
 * timeout. E' seguro porque cada sentido tem um unico escritor e um unico leitor, o
 * servidor manda cabecalho+payload num sendall so', e o firmware nunca deixa byte para
 * tras (qualquer divergencia fecha a conexao em vez de seguir lendo). */
static Exchange request_sample(uint32_t seq, float out[kAxes], WiFiClient &client)
{
  uint8_t request[5];
  request[0] = MSG_REQUEST;
  put_u32(request + 1, seq);
  if (!write_all(client, request, sizeof(request)))
  {
    return EXCHANGE_ERR_WRITE;
  }

  uint8_t header[5];
  if (!read_all(client, header, sizeof(header)))
  {
    return EXCHANGE_ERR_READ;
  }
  if (get_u32(header + 1) != seq) /* tripwire barato: fluxo dessincronizado */
  {
    return EXCHANGE_ERR_SEQ;
  }
  if (header[0] == MSG_END)
  {
    return EXCHANGE_END;
  }
  if (header[0] != MSG_SAMPLE)
  {
    return EXCHANGE_ERR_TYPE;
  }
  if (!read_all(client, (uint8_t *)out, kAxes * sizeof(float)))
  {
    return EXCHANGE_ERR_READ;
  }
  return EXCHANGE_OK;
}

/* A predicao da amostra `seq` e o ACK que confirma o log. Sem ACK a seq nao avanca:
 * adivinhar que a predicao chegou e' exatamente o que trocava real e pred de linha. */
static Exchange send_prediction(uint32_t seq, const result_t &result, WiFiClient &client)
{
  uint8_t msg[9];
  msg[0] = MSG_PREDICTION;
  put_u32(msg + 1, seq);
/* ESP32 e' little-endian como o PC: os 4 bytes do float saem na ordem que o
 * struct.unpack("<f") do servidor espera. */
#if defined(RF_MODEL_H) || defined(KNN_MODEL_H)
  float f_result = result;
#elif defined(MLP_MODEL_H)
  float f_result = result.classification[0].value;
#else
#error "Invalid model!"
#endif
  memcpy(msg + 5, &f_result, sizeof(float));
  if (!write_all(client, msg, sizeof(msg)))
  {
    return EXCHANGE_ERR_WRITE;
  }
  uint8_t response[5];
  if (!read_all(client, response, sizeof(response)))
  {
    return EXCHANGE_ERR_READ;
  }
  if (response[0] != MSG_ACK || get_u32(response + 1) != seq)
  {
    return EXCHANGE_ERR_ACK;
  }
  return EXCHANGE_OK;
}

/* A troca completa de UMA amostra, na MESMA conexao da anterior. */
static Exchange exchange_sample(void)
{
  float sample[kAxes];
  const Exchange outcome = request_sample(g_seq, sample, g_client);
  if (outcome != EXCHANGE_OK)
  {
    return outcome;
  }
  if (!runInference(sample, &g_result))
  {
    return EXCHANGE_ERR_INFERENCE;
  }
  return send_prediction(g_seq, g_result, g_client);
}
