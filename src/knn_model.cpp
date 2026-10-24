#include "knn_model.h"
#include <esp_timer.h>

static uint32_t s_inference_duration_us = 0UL;

bool knn_run_inference(const float *frame, knn_result_t *result)
{
  uint32_t inference_start = static_cast<uint32_t>(esp_timer_get_time());
  *result = knn_pca_prever(frame);
  s_inference_duration_us = static_cast<uint32_t>(esp_timer_get_time()) - inference_start;

  return true;
}

/* --------------------------------------------------------------------- debug */

void knn_print_result(const knn_result_t &result)
{
  Serial.print("  pred = ");
  Serial.print(result, 4);
  Serial.printf("   dsp %0.3fms", s_inference_duration_us / 1000UL);

#if defined(ARDUINO_ARCH_ESP32)
  /* Se este número cair ao longo de horas há fragmentação de heap; plano é ok. */
  Serial.print("  heap_livre ");
  Serial.print((uint32_t)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
#endif
  Serial.println();
}

void knn_print_banner()
{
  Serial.println();
  Serial.print("knob-senses: k-NN");
  Serial.print(" — janela ");
  Serial.print((int)kSamples);
  Serial.print("x");
  Serial.print((int)kAxes);
  Serial.print(" = ");
  Serial.print((int)kFrameSize);
  Serial.printf(" floats, %0.2f inferencia por segundo", 1000 / kPeriodMs);
  Serial.println();
  Serial.print("ordem dos eixos: ");
  Serial.println(KNOB_KNN_ORDEM);
  Serial.print("k = ");
  Serial.print(KNOB_KNN_K);
  Serial.print("   prototipos = ");
  Serial.println(KNOB_KNN_N_ITEMS);
}
