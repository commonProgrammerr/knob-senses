/*
 * knob-senses - Random Forest de regressao de PCA (potencia CA do inversor, kW).
 *
 * GERADO POR utils/treino_random_forest.ipynb. NAO EDITE A MAO: a proxima execucao
 * do notebook sobrescreve este arquivo.
 *
 * Como usar:
 *
 *     #include <knob_senses_rf.h>
 *
 *     const float bruto[KNOB_RF_N_FEATURES] = { 26.1f, 480.0f, 310.5f, 38.7f,
 *                                               sin1, cos1 };
 *     const float kw = pca_prever(bruto);
 *
 * ORDEM DAS GRANDEZAS BRUTAS:
 *   bruto[0] = temperatura_ambiente
 *   bruto[1] = ghit
 *   bruto[2] = jacques
 *   bruto[3] = sintetico
 *
 * O modelo foi treinado em z-scores e a normalizacao vive DENTRO desta lib (ver
 * knob_senses_rf.cpp): passe a grandeza fisica como ela veio do sensor. Nao normalize
 * antes de chamar pca_prever() - se o dado chegar em z-score, a conta acontece duas
 * vezes, tudo cai a ~7 desvios-padrao da media, e a previsao COLAPSA num valor fixo.
 *
 * Custo no ESP32-S3: 67.6 KB de flash, 0 de RAM. A floresta e' emitida como codigo
 * (emlearn method='inline'), nao como tabela - por isso nao ocupa RAM.
 */
#ifndef KNOB_SENSES_RF_H
#define KNOB_SENSES_RF_H

#include <stddef.h>

/* --------------------------------------------------------------- contrato */

#define KNOB_RF_N_FEATURES 4

/* Ordem esperada em bruto[]. As 4 primeiras sao exatamente a ordem que
 * utils/server.ipynb transmite pela rede, entao o firmware entrega o que
 * recebeu sem reordenar. */
#define KNOB_RF_ORDEM "temperatura_ambiente, ghit, jacques, sintetico"

/* Faixa fisica do alvo: o inversor satura em ~2 kW e o dataset nao tem zero (o
 * 0 era sentinela de faltante, removido na analise). A saida ja vem ceifada. */
#define KNOB_RF_PCA_MIN 0.01f
#define KNOB_RF_PCA_MAX 2.02f

/* ------------------------------------------------------------ procedencia */

#define KNOB_RF_N_ARVORES    30
#define KNOB_RF_FOLHA_MIN    50
#define KNOB_RF_N_NOS        4824
#define KNOB_RF_MAE_TESTE    0.1426f   /* kW, no split de teste (futuro) */
#define KNOB_RF_R2_TESTE     0.7732f

#ifdef __cplusplus
extern "C" {
#endif

/* bruto -> z-score -> floresta -> clip. Devolve a potencia CA em kW. */
float pca_prever(const float bruto[KNOB_RF_N_FEATURES]);

#ifdef __cplusplus
}
#endif

#endif /* KNOB_SENSES_RF_H */
