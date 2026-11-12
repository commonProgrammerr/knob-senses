/*
 * knob-senses - constantes da transformacao do MLP (potencia CA, kW).
 *
 * GERADO POR utils/gerar_header_mlp.py a partir de:
 *   dist/1h/normalizacao.json
 *   dist/1h/transformacao_alvo.json
 * NAO EDITE A MAO: a proxima geracao sobrescreve este arquivo.
 *
 * Ordem do pos-processamento:
 *   clip_boxcox -> inv_boxcox -> - shift_kw -> + offset_kw
 *   -> clip(pca_min_kw, pca_max_kw)
 */
#ifndef KNOB_MLP_TRANSFORMACAO_ALVO_H
#define KNOB_MLP_TRANSFORMACAO_ALVO_H

#include <stddef.h>

constexpr size_t KNOB_MLP_N_FEATURES = 3;
#define KNOB_MLP_ORDEM "temp_amb, ghit, jacques"
#define KNOB_MLP_FREQ  "1h"

/* Min-max ajustado SO no treino (2021..2024H1), na ordem acima. */
constexpr float KNOB_MLP_MINIMO[KNOB_MLP_N_FEATURES] = {
  15.6436667f,
  5.52674211f,
  17.6906019f
};
constexpr float KNOB_MLP_MAXIMO[KNOB_MLP_N_FEATURES] = {
  38.2178333f,
  1274.00339f,
  67.3596957f
};

/* Shift + Box-Cox, ajustados SO no treino. */
constexpr float KNOB_MLP_SHIFT_KW        = 0.001f;
constexpr float KNOB_MLP_BOXCOX_LAMBDA   = 0.651434136f;
constexpr float KNOB_MLP_BOXCOX_CLIP_MIN = -1.51802057f;
constexpr float KNOB_MLP_BOXCOX_CLIP_MAX = 0.849329283f;

/* Calibracao de vies (2024H2) e faixa fisica do alvo. */
constexpr float KNOB_MLP_OFFSET_KW       = -0.0593990355f;
constexpr float KNOB_MLP_PCA_MIN_KW      = 0.0f;
constexpr float KNOB_MLP_PCA_MAX_KW      = 2.0f;

static_assert(1.0f + KNOB_MLP_BOXCOX_LAMBDA * KNOB_MLP_BOXCOX_CLIP_MIN > 0.0f,
              "clip_boxcox inferior fora do dominio da inversa");
static_assert(1.0f + KNOB_MLP_BOXCOX_LAMBDA * KNOB_MLP_BOXCOX_CLIP_MAX > 0.0f,
              "clip_boxcox superior fora do dominio da inversa");

#endif /* KNOB_MLP_TRANSFORMACAO_ALVO_H */
