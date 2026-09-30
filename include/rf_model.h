#ifndef RF_MODEL_H
#define RF_MODEL_H

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <HardwareSerial.h>

#include "knob_senses_rf.h"

extern HardwareSerial Serial;

constexpr size_t kSamples = 1U;       
constexpr size_t kAxes = KNOB_RF_N_FEATURES;
constexpr size_t kFrameSize = KNOB_RF_N_FEATURES;
constexpr uint32_t kPeriodMs = 25U;       

typedef float rf_result_t;

bool rf_run_inference(const float *frame, rf_result_t *result);

void rf_print_result(const rf_result_t &result);

void rf_print_banner();

#endif