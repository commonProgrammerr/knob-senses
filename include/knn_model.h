#ifndef KNN_MODEL_H
#define KNN_MODEL_H

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <HardwareSerial.h>

#include "knob_senses_knn.h"

extern HardwareSerial Serial;

constexpr size_t kSamples = 1U;
constexpr size_t kAxes = KNOB_KNN_N_FEATURES;
constexpr size_t kFrameSize = KNOB_KNN_N_FEATURES;
constexpr uint32_t kPeriodMs = 10U;

typedef float knn_result_t;

bool knn_run_inference(const float *frame, knn_result_t *result);

void knn_print_result(const knn_result_t &result);

void knn_print_banner();

#endif
