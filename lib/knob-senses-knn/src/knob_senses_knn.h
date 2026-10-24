/*
 * knob-senses - k-NN (emlearn) de regressao de PCA (potencia CA do inversor, kW).
 *
 * GERADO POR utils/treino_knn.ipynb. NAO EDITE A MAO: a proxima execucao do notebook
 * sobrescreve este arquivo.
 *
 * Como usar:
 *
 *     #include <knob_senses_knn.h>
 *
 *     const float bruto[KNOB_KNN_N_FEATURES] = { 480.0f, 310.5f, 26.1f };
 *     const float kw = knn_pca_prever(bruto);
 *
 * ORDEM DAS GRANDEZAS BRUTAS:
 *   bruto[0] = temperatura_ambiente
 *   bruto[1] = ghit
 *   bruto[2] = jacques
 *   bruto[3] = sintetico
 *
 * O z-score vive DENTRO desta lib e tudo e' quantizado em int16 (imposicao do emlearn):
 * z vira lrintf(z * 1000.0) com clamp em +-8.0f (o clamp garante que o
 * `eml_isqrt` do emlearn nao trave), e o alvo vira miliwatts. A saida e' a MEDIA
 * inteira dos 3 vizinhos mais proximos, de volta para kW e ceifada em
 * [0.01, 2.02].
 *
 * Custo no ESP32-S3: ~9.8 KB de flash (dados + rotulos int16) e ~7.8 KB
 * de RAM (scratch do emlearn: 8 B por prototipo) - o numero de prototipos e' o botao.
 */
#ifndef KNOB_SENSES_KNN_H
#define KNOB_SENSES_KNN_H

#include <stddef.h>

/* --------------------------------------------------------------- contrato */

#define KNOB_KNN_N_FEATURES 4

/* Ordem esperada em bruto[] - a mesma que o servidor de teste transmite. */
#define KNOB_KNN_ORDEM "temperatura_ambiente, ghit, jacques, sintetico"

/* Faixa fisica do alvo (a saida ja' vem ceifada). */
#define KNOB_KNN_PCA_MIN 0.01f
#define KNOB_KNN_PCA_MAX 2.02f

/* ------------------------------------------------------------- k-NN + quantizacao */

#define KNOB_KNN_N_ITEMS 1000   /* prototipos guardados (flash E RAM = 8 x isto) */
#define KNOB_KNN_K 3               /* vizinhos da media */
#define KNOB_KNN_Z_CLAMP 8.0f
#define KNOB_KNN_Z_SCALE 1000.0f
#define KNOB_KNN_Y_SCALE 1000.0f

/* ------------------------------------------------------------ procedencia */

#define KNOB_KNN_MAE_TESTE 0.1476f   /* kW, no split de teste (futuro) */
#define KNOB_KNN_R2_TESTE 0.7669f

#ifdef __cplusplus
extern "C" {
#endif

/* bruto -> z/clamp/int16 -> kNN do emlearn -> media de k -> clip. kW. */
float knn_pca_prever(const float bruto[KNOB_KNN_N_FEATURES]);

#ifdef __cplusplus
}
#endif

#endif /* KNOB_SENSES_KNN_H */
