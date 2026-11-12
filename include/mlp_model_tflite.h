#ifndef MLP_MODEL_TFLITE_H
#define MLP_MODEL_TFLITE_H

#include <stddef.h>
#include <stdint.h>

constexpr size_t kSamples = 3;
constexpr size_t kAxes = 1;
constexpr size_t kFrameSize = 3;
constexpr uint32_t kPeriodMs = 1;

bool mlp_run_inference(const float *frame, float *out);
void mlp_print_result(const float &value);
void mlp_print_banner();

#endif
