#include "mlp_model_tflite.h"

#include <Arduino.h>

#include "knob_mlp_model.h"
#include "mlp_posprocessamento.h"

static_assert(KNOB_MLP_N_FEATURES == kFrameSize,
              "o header de transformacao nao cobre o frame do modelo");

bool mlp_run_inference(const float *frame, float *out)
{
  float normalizado[KNOB_MLP_N_FEATURES];
  mlp_normalizar(frame, normalizado);

  float y_bc = 0.0f;
  if (!knob_mlp_run(normalizado, &y_bc))
  {
    Serial.print("erro no TFLite Micro: ");
    Serial.println(knob_mlp_last_error());
    return false;
  }

  *out = mlp_posprocessar_saida(y_bc);

#if defined(KNOB_MLP_DEBUG)
  static uint32_t linhas_dbg = 0;
  if (linhas_dbg < 8)
  {
    Serial.printf(
        "dbg %lu | bruto %.4f %.4f %.4f | norm %.5f %.5f %.5f | "
        "bc %.6f | kw %.4f\n",
        (unsigned long)linhas_dbg,
        frame[0], frame[1], frame[2],
        normalizado[0], normalizado[1], normalizado[2],
        y_bc, *out);
    linhas_dbg++;
  }
#endif
  return true;
}

void mlp_print_result(const float &value)
{
  Serial.print("  pred = ");
  Serial.println(value, 4);
}

void mlp_print_banner()
{
  Serial.println();
  Serial.print("knob-senses MLP TFLite Micro - janela ");
  Serial.print((int)kSamples);
  Serial.print("x");
  Serial.print((int)kAxes);
  Serial.print(" = ");
  Serial.print((int)kFrameSize);
  Serial.println(" floats");
  Serial.println();
}
