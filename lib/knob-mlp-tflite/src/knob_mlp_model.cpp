#include "knob_mlp_model.h"

#include <math.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#if defined(KNOB_TFLITE_MICRO_EXTERNAL)
#include "tensorflow/lite/micro/schema/schema_generated.h"
#else
#include "tensorflow/lite/schema/schema_generated.h"
#endif

namespace
{
const tflite::Model *g_model = nullptr;
tflite::MicroMutableOpResolver<1> g_resolver;
alignas(16) uint8_t g_arena[kKnobMlpTensorArenaSize];
tflite::MicroInterpreter *g_interpreter = nullptr;
const char *g_error = nullptr;
bool g_pronto = false;
} // namespace

bool knob_mlp_init()
{
  if (g_pronto)
  {
    return true;
  }

  g_model = tflite::GetModel(g_knob_mlp_model);
  if (g_model->version() != TFLITE_SCHEMA_VERSION)
  {
    g_error = "schema do modelo";
    return false;
  }

  g_resolver.AddFullyConnected();

  static tflite::MicroInterpreter interpreter(
      g_model, g_resolver, g_arena, sizeof(g_arena));
  g_interpreter = &interpreter;

#if defined(KNOB_TFLITE_MICRO_EXTERNAL)
  if (interpreter.AllocateTensors() != kTfLiteOk)
#else
  if (interpreter.AllocateTensors(true) != kTfLiteOk)
#endif
  {
    g_error = "arena de tensores";
    return false;
  }

  TfLiteTensor *input = interpreter.input(0);
  if (input == nullptr ||
      (input->type != kTfLiteFloat32 && input->type != kTfLiteInt8) ||
      input->dims->size != 2 || input->dims->data[1] != (int)kKnobMlpInputFeatures)
  {
    g_error = "tensor de entrada";
    return false;
  }

  g_pronto = true;
  return true;
}

bool knob_mlp_run(const float *normalized, float *bc_out)
{
  if (!g_pronto && !knob_mlp_init())
  {
    return false;
  }

  TfLiteTensor *input = g_interpreter->input(0);
  if (input->type == kTfLiteInt8)
  {
    const float scale = input->params.scale;
    const int zero = input->params.zero_point;
    for (size_t i = 0; i < kKnobMlpInputFeatures; i++)
    {
      int q = (int)lrintf(normalized[i] / scale) + zero;
      q = q < -128 ? -128 : (q > 127 ? 127 : q);
      input->data.int8[i] = (int8_t)q;
    }
  }
  else
  {
    for (size_t i = 0; i < kKnobMlpInputFeatures; i++)
    {
      input->data.f[i] = normalized[i];
    }
  }

  if (g_interpreter->Invoke() != kTfLiteOk)
  {
    g_error = "invoke";
    return false;
  }

  const TfLiteTensor *output = g_interpreter->output(0);
  if (output->type == kTfLiteInt8)
  {
    *bc_out = (output->data.int8[0] - output->params.zero_point) *
              output->params.scale;
  }
  else
  {
    *bc_out = output->data.f[0];
  }
  return true;
}

const char *knob_mlp_last_error()
{
  return g_error;
}
