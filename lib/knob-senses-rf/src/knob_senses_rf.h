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
 *     const float bruto[KNOB_RF_N_FEATURES] = { 26.1f, 480.0f, 38.7f };
 *     const float kw = pca_prever(bruto);
 *
 * ORDEM DAS GRANDEZAS BRUTAS:
 *   bruto[0] = temperatura_ambiente
 *   bruto[1] = ghit
 *   bruto[2] = jacques
 *
 * O modelo foi treinado em z-scores e a normalizacao vive DENTRO desta lib (ver
 * knob_senses_rf.cpp): passe a grandeza fisica como ela veio do sensor. Nao normalize
 * antes de chamar pca_prever() - se o dado chegar em z-score, a conta acontece duas
 * vezes, tudo cai a ~7 desvios-padrao da media, e a previsao COLAPSA num valor fixo.
 *
 * Custo: a floresta e' emitida como codigo (emlearn method='inline'), entao nao
 * ocupa RAM; o flash depende do tamanho da floresta - meca no objeto.
 */
#ifndef KNOB_SENSES_RF_H
#define KNOB_SENSES_RF_H

#include <stddef.h>

/* --------------------------------------------------------------- contrato */

#define KNOB_RF_N_FEATURES 3

/* Ordem esperada em bruto[]. As 4 primeiras sao exatamente a ordem que
 * utils/server.ipynb transmite pela rede, entao o firmware entrega o que
 * recebeu sem reordenar. */
#define KNOB_RF_ORDEM "temperatura_ambiente, ghit, jacques"

/* Faixa fisica do alvo: 0..2 kW. O inversor nao devolve potencia negativa e
 * satura perto de 2 kW; a saida ja vem ceifada. */
#define KNOB_RF_PCA_MIN 0.0f
#define KNOB_RF_PCA_MAX 2.0f

/* ------------------------------------------------------------ procedencia */

#define KNOB_RF_N_ARVORES    12
#define KNOB_RF_FOLHA_MIN    200
#define KNOB_RF_N_NOS        4532
#define KNOB_RF_MAE_TESTE    0.1137f   /* kW, no split de teste (futuro) */
#define KNOB_RF_R2_TESTE     0.9109f

/* Calibracao de vies: somada antes do clip. Foi ajustada na janela de
 * calibracao (2024H2), nunca no teste, para corrigir a deriva de eficiencia
 * 2024->2025. Se o modelo for retreinado, recalcule-a no notebook. */
#define KNOB_RF_OFFSET       -0.099050f

#ifdef __cplusplus
extern "C" {
#endif

/* bruto -> z-score -> floresta -> offset -> clip. Devolve a potencia CA em kW. */
float pca_prever(const float bruto[KNOB_RF_N_FEATURES]);

#ifdef __cplusplus
}
#endif

#endif /* KNOB_SENSES_RF_H */
