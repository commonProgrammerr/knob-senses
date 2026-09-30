#include <Arduino.h>
#include <string.h>
#include <WiFi.h>
#include <WiFiClient.h>

#include "rf_model.h"

#define SERVER_IP "192.168.0.14"
#define SERVER_PORT 3333
#define WIFI_SSID "90225"
#define WIFI_PASSWORD "Segred0%%@2444"

/* --------------------------------------------------------------- estado */

#if defined(RF_MODEL_H)

#define runInference(frame, result) rf_run_inference(frame, result)
#define printResult(result) rf_print_result(result)
#define printBanner() rf_print_banner()
rf_result_t g_result;
typedef rf_result_t result_t;

#elif defined(MLP_MODEL_H)

#define runInference(frame, result) mlp_run_inference(frame, result)
#define printResult(result) mlp_print_result(result)
#define printBanner() mlp_print_banner()
ei_impulse_result_t g_result;
typedef ei_impulse_result_t result_t;

#else
  #error "Invalid model!"
#endif

static uint32_t g_nextDueMs = 0;
static uint32_t g_sampleSeq = 0; // amostras capturadas; usado pela demo
static uint32_t g_overruns = 0;  // ticks em que atrasamos um período inteiro
static uint32_t g_dropouts = 0;  // ticks sem amostra disponível
static bool g_inferenceFailed = false;

WiFiClient g_client;

static bool get_sample_data(float out[kAxes], WiFiClient &client);
static void send_inference_results(const result_t &result, WiFiClient &client);

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

void setup()
{
  ledSetup();

  Serial.begin(115200);

  printBanner();

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
  if ((uint32_t)(now - g_nextDueMs) < kPeriodMs)
  {
    return;
  }

  g_nextDueMs += kPeriodMs;
  if ((uint32_t)(millis() - g_nextDueMs) >= kPeriodMs)
  {
    g_overruns++; /* atrasamos um período inteiro: nao tenta recuperar em rajada */
    g_nextDueMs = millis();
  }


  float sample[kAxes];
  if (!get_sample_data(sample, g_client))
  {
    g_dropouts++;
    Serial.print("sem amostra (dropout ");
    Serial.print(g_dropouts);
    Serial.println(")");
    return;
  }

  if (g_inferenceFailed)
  {
    return;
  }

  ledWrite(true);


  if (!runInference(sample, &g_result))
  {
    g_inferenceFailed = true;
    Serial.println("ERRO FATAL na inferencia - parando de inferir");
    Serial.println("(insistir vazaria a arena do TFLite a cada segundo)");
    ledWrite(false);
    return;
  }

  printResult(g_result);
  send_inference_results(g_result, g_client);
  ledWrite(false);
  g_client.stop();
}

static bool get_sample_data(float out[kAxes], WiFiClient &client)
{
  client.connect(SERVER_IP, SERVER_PORT);
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

static void send_inference_results(const result_t &result, WiFiClient &client)
{
  client.connect(SERVER_IP, SERVER_PORT);
  if (!client.connected())
  {
    Serial.println("Falha na conexão com o servidor");
    return;
  }
  size_t total_request_size = sizeof(float) + 1U;
  const uint8_t *send_buf = (const uint8_t *)&(result);

  size_t bytesWritten = (client.write((const uint8_t *)"\x02", 1U) +
                         client.write(send_buf, sizeof(float))); // send body

  if (bytesWritten != sizeof(float) + 1U)
  {
    Serial.println("Falha ao enviar resultados para o servidor");
    Serial.printf("Bytes enviados/esperados: %U/%U", bytesWritten, total_request_size);
  }
}
