#ifndef MLP_MODEL_H
#define MLP_MODEL_H

#include <string.h>
#include <HardwareSerial.h>
/* So' TIPOS e macros do modelo (ei_impulse_result_t + EI_CLASSIFIER_*). Este header
 * tambem entra no main.cpp, e o SDK do Edge Impulse define funcoes/variaveis (e o
 * proprio .tflite, via asm .incbin) nos headers de implementacao: incluir o
 * ei_run_classifier.h aqui duplicaria tudo em DOIS TUs (main.cpp e mlp_model.cpp) e o
 * link morre com "multiple definition". O unico TU com o SDK completo e' o
 * mlp_model.cpp — manter esta separacao e' o que faz o link fechar. */
#include <edge-impulse-sdk/classifier/ei_classifier_types.h>

extern HardwareSerial Serial;

/* Tudo derivado do modelo — nunca hardcode estes números. */
constexpr size_t kSamples = EI_CLASSIFIER_RAW_SAMPLE_COUNT;       // 5
constexpr size_t kAxes = EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME;     // 9
constexpr size_t kFrameSize = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; // 45
constexpr uint32_t kPeriodMs = EI_CLASSIFIER_INTERVAL_MS;         // 1000

static_assert(kFrameSize == kSamples * kAxes,
              "janela do impulso nao bate com RAW_SAMPLE_COUNT * RAW_SAMPLES_PER_FRAME");
static_assert(EI_CLASSIFIER_LABEL_COUNT == 1, "esperada 1 saida de regressao");


extern ei_impulse_result_t g_result;

bool mlp_run_inference(const float *frame, ei_impulse_result_t *result);

void mlp_print_result(const ei_impulse_result_t &result);

void mlp_print_banner();

#endif