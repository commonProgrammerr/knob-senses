#include <Arduino.h>
#include <string.h>
#include <HWCDC.h>     /* a CDC do USB-Serial/JTAG (porta "USB" nativa) carrega os dados */

/* O modelo ativo vem do ENV do PlatformIO (platformio.ini): um env por modelo, com as
 * flags especificas isoladas (ex.: o Edge Impulse so' no mlp). Nao troque o include a
 * mao - escolha o env:  pio run -e rf | -e mlp  */
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

#else
#error "Nenhum modelo: use um env do platformio.ini (-e rf | -e mlp)"
#endif

/* ---------------------------------------------------- protocolo v3 (lotes)
 *
 * O transporte do ensaio e' a CDC do USB-Serial/JTAG (COM29); o log de execucao vai
 * pela UART0 (COM28). O host manda LOTES de amostras e recebe LOTES de predicoes:
 * UM par de transferencias por lote, em vez de duas por linha.
 *
 *   lote (host -> ESP):   0x01 + base u32 + count u32 + count x 3 float32
 *   resposta (ESP -> host): 0x03 + base u32 + count u32 + count x float32
 *
 * Menos mensagens = menos exposicao a soluços do elo. Medido nesta bancada: o
 * downlink (host -> ESP) deste par de driver/placa tem soluços de ~2 s a cada ~30
 * mensagens pequenas; com o protocolo linha a linha, cada soluço envenenava a troca
 * seguinte (o ACK duplicado do reenvio chegava atrasado e virava o cabecalho da
 * proxima amostra). Em lote, um soluço custa UM reenvio por ~64 linhas.
 *
 * O firmware e' APATRIDA por lote: nao ha' seq nem retentativa. Quem controla o
 * andamento e' o host - ele reenvia o MESMO lote se a resposta nao chegar, e o
 * reprocessamento e' inofensivo (mesmas amostras, mesmas predicoes). O handshake
 * continua com as duas magias: RESET (0x52) e' o primeiro de cada serve_serial
 * (nova usina); RESUME (0x4B) e' a reentrada apos um soluço. O dreno pos-MAGIC
 * descarta as copias velhas do MAGIC que o host reenvia ate' a resposta chegar.
 */
#define MSG_BATCH 0x01        /* host -> ESP: cabecalho do lote de amostras        */
#define MSG_BATCH_REPLY 0x03  /* ESP -> host: cabecalho do lote de predicoes       */
#define MSG_MAGIC 0x4B        /* 'K': RESUME - host -> ESP; e o ACK do handshake   */
#define MSG_MAGIC_RESET 0x52  /* 'R': RESET - host -> ESP                          */
#define BATCH_MAX 64U         /* maximo de amostras por lote (payload 768 B)       */

/* Tempo maximo POR BYTE numa leitura, em SEGUNDOS (a unidade de negocio; o
 * Stream::setTimeout quer MILISSEGUNDOS - ver setup()). O host aguenta 5 s: o
 * firmware desiste PRIMEIRO e volta a aguardar o MAGIC. */
#define SERVER_TIMEOUT_S 2U

/* ------------------------------------------------------------------ estado */

static bool g_session = false;  /* true entre o MAGIC e o fim/erro da sessao */
static bool g_serverOk = true;  /* false apos uma falha; true de proposito, para o
                                 * primeiro "servidor fora do ar" imprimir a transicao */
static uint32_t g_dropouts = 0; /* falhas de sessao; o host reenvia o lote, nada se perde */
static bool g_inferenceFailed = false;
result_t g_result;

/* Diagnostico do lote: acumulados impressos a cada 10 lotes. */
static uint32_t g_diag_lotes = 0;
static uint32_t g_diag_ms = 0;

static void record_failure(const char *reason);
static void record_success(void);
static uint8_t await_magic(HWCDC &s);
static bool write_all(Stream &client, const uint8_t *buf, size_t len);
static bool read_all(Stream &client, uint8_t *buf, size_t len);
static void put_u32(uint8_t *p, uint32_t v);
static uint32_t get_u32(const uint8_t *p);

/* Le o byte magico do host e devolve QUAL magia veio (0 = nada ainda). O ROM
 * bootloader escreve na CDC antes de nos: bytes anteriores ao MAGIC sao lixo de
 * boot, nao protocolo. */
static uint8_t await_magic(HWCDC &s)
{
  while (s.available() > 0)
  {
    const int b = s.read();
    if (b == MSG_MAGIC || b == MSG_MAGIC_RESET)
    {
      return (uint8_t)b;
    }
  }
  return 0;
}

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

  Serial.begin(115200);   /* log na UART0 (CH343, COM28) */

  printBanner();

  /* Dados do ensaio na CDC do USB-Serial/JTAG (COM29). O baud e' virtual em USB:
   * so' precisa ser valido; a taxa real e' a do barramento. */
  USBSerial.begin(115200);
  /* Um lote cheio sao 777 bytes: a fila de RX padrao (256 B) nao comporta. */
  USBSerial.setRxBufferSize(1024);
  /* Stream::setTimeout e' em MILISSEGUNDOS (o WiFiClient e' que recebia segundos). */
  USBSerial.setTimeout(SERVER_TIMEOUT_S * 1000U);
}

/* -------------------------------------------------------------------- loop */

void loop()
{
  /* Sem gate de ritmo: o ensaio anda na velocidade do lote (o kPeriodMs dos
   * cabecalhos e' o ritmo da inferencia em tempo real, nao do ensaio). A sessao
   * comeca com o MAGIC do host e re-arma sozinha apos fim/erro. */
  if (g_inferenceFailed)
  {
    return;
  }
  if (!g_session)
  {
    const uint8_t magia = await_magic(USBSerial);
    if (magia == 0)
    {
      return;
    }
    /* O host reenvia o MAGIC ate' a resposta chegar: a fila pode ter VARIOS
     * magicos atras do primeiro, e cada um envenenaria o primeiro lote. Esvazia
     * tudo: antes do MAGIC_ACK o host nao manda nada que preste. */
    while (USBSerial.available() > 0)
    {
      USBSerial.read();
    }
    USBSerial.write(MSG_MAGIC); /* MAGIC de volta: daqui em diante so' protocolo */
    if (magia == MSG_MAGIC_RESET)
    {
      Serial.println("sessao serial iniciada (RESET)");
    }
    else
    {
      Serial.println("sessao serial retomada (RESUME)");
    }
    g_session = true;
  }

  const uint32_t t0 = millis();

  /* Cabecalho do lote: tipo + base + count. */
  uint8_t header[9];
  if (!read_all(USBSerial, header, sizeof(header)))
  {
    record_failure("cabecalho do lote");
    g_session = false;
    return;
  }
  if (header[0] != MSG_BATCH)
  {
    record_failure("tipo do lote");
    g_session = false;
    return;
  }
  const uint32_t base = get_u32(header + 1);
  const uint32_t count = get_u32(header + 5);
  if (count == 0)
  {
    g_session = false; /* fim do dataset: o host encerrou; volta a aguardar o MAGIC */
    Serial.println("fim do dataset (lote vazio)");
    return;
  }
  if (count > BATCH_MAX)
  {
    record_failure("tamanho do lote");
    g_session = false;
    return;
  }

  /* Payload: count amostras de kFrameSize floats, em grandeza bruta. */
  float batch[BATCH_MAX][kFrameSize];
  for (uint32_t i = 0; i < count; i++)
  {
    if (!read_all(USBSerial, (uint8_t *)batch[i], kFrameSize * sizeof(float)))
    {
      record_failure("payload do lote");
      g_session = false;
      return;
    }
  }

  /* Inferencia do lote inteiro e a resposta numa escrita so'. */
  static uint8_t reply[9 + BATCH_MAX * sizeof(float)];
  reply[0] = MSG_BATCH_REPLY;
  put_u32(reply + 1, base);
  put_u32(reply + 5, count);
  for (uint32_t i = 0; i < count; i++)
  {
    ledWrite(true);
    const bool ok = runInference(batch[i], &g_result);
    ledWrite(false);
    if (!ok)
    {
      g_inferenceFailed = true;
      Serial.println("ERRO FATAL na inferencia - parando de inferir");
      Serial.println("(insistir vazaria a arena do TFLite a cada segundo)");
      g_session = false;
      return;
    }
#if defined(RF_MODEL_H)
    float f_result = g_result;
#elif defined(MLP_MODEL_H)
    float f_result = g_result.classification[0].value;
#else
#error "Invalid model!"
#endif
    memcpy(reply + 9 + i * sizeof(float), &f_result, sizeof(float));
  }
  if (!write_all(USBSerial, reply, 9 + count * sizeof(float)))
  {
    record_failure("envio da resposta");
    g_session = false;
    return;
  }

  record_success();
  g_diag_lotes++;
  g_diag_ms += millis() - t0;
  if ((g_diag_lotes % 10U) == 0U)
  {
    Serial.print("lotes: ");
    Serial.print(g_diag_lotes);
    Serial.printf(" | media %lu ms/lote | ultimo base=%lu count=%lu\n",
                  (unsigned long)(g_diag_ms / g_diag_lotes),
                  (unsigned long)base, (unsigned long)count);
  }
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

/* write() devolve o que conseguiu enviar - pode ser PARCIAL - e devolve 0 se o
 * transporte caiu. Mensagem so' sai inteira com este laco; parcial e' falha. */
static bool write_all(Stream &client, const uint8_t *buf, size_t len)
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
static bool read_all(Stream &client, uint8_t *buf, size_t len)
{
  return client.readBytes((char *)buf, len) == len;
}

/* Falha = o host reenviara' o MESMO lote no proximo MAGIC: conta e imprime SO' a
 * transicao - na velocidade do ensaio, um print por falha vira parede de log. */
static void record_failure(const char *reason)
{
  g_dropouts++;
  if (g_serverOk)
  {
    Serial.print("lote falhou (");
    Serial.print(reason);
    Serial.print(") - o host reenviara' o MESMO lote. falhas=");
    Serial.println(g_dropouts);
    g_serverOk = false;
  }
}

/* O lote saiu inteiro: a sessao esta' viva. */
static void record_success(void)
{
  if (!g_serverOk)
  {
    Serial.print("servidor respondeu de novo (falhas ate agora: ");
    Serial.print(g_dropouts);
    Serial.println(")");
    g_serverOk = true;
  }
}
