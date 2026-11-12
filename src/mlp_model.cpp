#include "mlp_model.h"

/* O UNICO TU com o SDK de implementacao do Edge Impulse (ver o comentario no
 * mlp_model.h): run_classifier(), numpy::signal_from_buffer() e o modelo em si. */
#include <edge-impulse-sdk/classifier/ei_run_classifier.h>

#include "mlp_posprocessamento.h"

static_assert(KNOB_MLP_N_FEATURES == kFrameSize,
              "o header de transformacao nao cobre o frame do impulso");

bool mlp_run_inference(const float *frame, ei_impulse_result_t *result)
{
  float normalizado[KNOB_MLP_N_FEATURES];
  mlp_normalizar(frame, normalizado);

  signal_t signal;
  if (numpy::signal_from_buffer(normalizado, kFrameSize, &signal) != EIDSP_OK)
  {
    Serial.println("erro: nao foi possivel montar o signal");
    return false;
  }

  /* process_impulse() ja zera o result; nao precisa memset aqui. */
  if (run_classifier(&signal, result, false) != EI_IMPULSE_OK)
  {
    return false;
  }

  /* A partir daqui o valor esta em kW: print e lote leem o mesmo campo. */
  result->classification[0].value =
      mlp_posprocessar_saida(result->classification[0].value);
  return true;
}

/* --------------------------------------------------------------------- debug */

void mlp_print_result(const ei_impulse_result_t &result)
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
  Serial.println();
}

void mlp_print_banner()
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
  Serial.printf(" floats, %0.2f inferencia por segundo\n\r", 1000U / kPeriodMs);
  Serial.print("ordem dos eixos: ");
  Serial.println(EI_CLASSIFIER_FUSION_AXES_STRING);
  Serial.println();
}
