#include "mlp_model.h"

/* O UNICO TU com o SDK de implementacao do Edge Impulse (ver o comentario no
 * mlp_model.h): run_classifier(), numpy::signal_from_buffer() e o modelo em si. */
#include <edge-impulse-sdk/classifier/ei_run_classifier.h>

#include "mlp_posprocessamento.h"

#include <math.h>

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

  const float y_bc = result->classification[0].value;
  /* A partir daqui o valor esta em kW: print e lote leem o mesmo campo. */
  result->classification[0].value = mlp_posprocessar_saida(y_bc);

#if defined(KNOB_MLP_DEBUG)
  static uint32_t linhas_dbg = 0;
  if (linhas_dbg < 8)
  {
    /* Parametros do modelo int8 de 1 h (modelo_quantizado.tflite). */
    constexpr float in_scale = 0.004108027555048466f;
    constexpr int in_zero = -128;
    constexpr float out_scale = 0.008816505782306194f;
    constexpr int out_zero = 28;
    auto quantizar = [](float v, float s, int z) {
      return (int)lrintf(v / s) + z;
    };
    Serial.printf(
        "dbg %lu | bruto %.4f %.4f %.4f | norm %.5f %.5f %.5f | "
        "q %d %d %d | bc %.6f | qout %d | kw %.4f\n",
        (unsigned long)linhas_dbg,
        frame[0], frame[1], frame[2],
        normalizado[0], normalizado[1], normalizado[2],
        quantizar(normalizado[0], in_scale, in_zero),
        quantizar(normalizado[1], in_scale, in_zero),
        quantizar(normalizado[2], in_scale, in_zero),
        y_bc,
        quantizar(y_bc, out_scale, out_zero),
        result->classification[0].value);
    linhas_dbg++;
  }
#endif
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
