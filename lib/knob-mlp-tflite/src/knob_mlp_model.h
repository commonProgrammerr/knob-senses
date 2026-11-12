#ifndef KNOB_MLP_MODEL_H
#define KNOB_MLP_MODEL_H

#include <stddef.h>
#include <stdint.h>

extern const unsigned char g_knob_mlp_model[];
extern const unsigned int g_knob_mlp_model_len;

constexpr size_t kKnobMlpInputFeatures = 3;
constexpr size_t kKnobMlpTensorArenaSize = 8192;

bool knob_mlp_init();
bool knob_mlp_run(const float *normalized, float *bc_out);
const char *knob_mlp_last_error();

#endif
