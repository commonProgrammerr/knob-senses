#ifndef MLP_POSPROCESSAMENTO_H
#define MLP_POSPROCESSAMENTO_H

#include <math.h>
#include <stddef.h>

#include "mlp_transformacao_alvo.h"

/* Min-max SEM clamp: a quantizacao int8 do TFLite satura em [0, 1.0475].
 * O Python de treino tambem nao clampava; manter a mesma conta evita
 * divergencia entre a avaliacao offline e a placa. */
inline void mlp_normalizar(const float *bruto, float *normalizado)
{
  for (size_t i = 0; i < KNOB_MLP_N_FEATURES; i++)
  {
    normalizado[i] = (bruto[i] - KNOB_MLP_MINIMO[i]) /
                     (KNOB_MLP_MAXIMO[i] - KNOB_MLP_MINIMO[i]);
  }
}

/* Inversa de Box-Cox. Lambda == 0 vira expf; o caso normal e' powf. */
inline float mlp_boxcox_inversa(float z)
{
  const float lambda = KNOB_MLP_BOXCOX_LAMBDA;
  if (lambda == 0.0f)
  {
    return expf(z);
  }

  const float base = 1.0f + lambda * z;
  if (base <= 0.0f)
  {
    /* Defensivo: as constantes geradas mantem base > 0. */
    return 0.0f;
  }
  return powf(base, 1.0f / lambda);
}

/* Ordem: clip_boxcox -> inv_boxcox -> - shift_kw -> + offset_kw -> clip fisico. */
inline float mlp_posprocessar_saida(float y_bc)
{
  float z = y_bc;
  // if (z < KNOB_MLP_BOXCOX_CLIP_MIN) z = KNOB_MLP_BOXCOX_CLIP_MIN;
  // if (z > KNOB_MLP_BOXCOX_CLIP_MAX) z = KNOB_MLP_BOXCOX_CLIP_MAX;

  float y = mlp_boxcox_inversa(z);
  y -= KNOB_MLP_SHIFT_KW;
  y += KNOB_MLP_OFFSET_KW;

  if (y < KNOB_MLP_PCA_MIN_KW) y = KNOB_MLP_PCA_MIN_KW;
  if (y > KNOB_MLP_PCA_MAX_KW) y = KNOB_MLP_PCA_MAX_KW;
  return y;
}

#endif /* MLP_POSPROCESSAMENTO_H */
