#include "knob_senses_rf.h"

/* As arvores, geradas pelo emlearn. Este header tem arrays static e so depende de
 * <stdint.h>; inclua-o em UM unico .cpp, senao a tabela entra duplicada no binario
 * (sem erro de link, mas com flash desperdicado). */
#include "modelo_pca.h"

/* Media e desvio, ajustados SO no split de treino (nunca na validacao nem no
 * teste). Recalculados do bruto a cada execucao de utils/treino_random_forest.ipynb
 * e conferidos contra data/processed/*.csv - ver a secao 2 do notebook. */
static const float PCA_MEDIA[KNOB_RF_N_FEATURES] = {
    26.887260f,
    470.118520f,
    39.087056f,
    -0.005921f,
    0.024009f,
    0.846211f,
    0.033089f
};

static const float PCA_DESVIO[KNOB_RF_N_FEATURES] = {
    4.082861f,
    288.184946f,
    11.100748f,
    0.706457f,
    0.707324f,
    0.549053f,
    0.178868f
};

float pca_prever(const float bruto[KNOB_RF_N_FEATURES])
{
    float z[KNOB_RF_N_FEATURES];
    for (size_t i = 0; i < KNOB_RF_N_FEATURES; i++) {
        z[i] = (bruto[i] - PCA_MEDIA[i]) / PCA_DESVIO[i];
    }

    float p = modelo_pca_predict(z, KNOB_RF_N_FEATURES);

    /* A floresta extrapola fora do que viu; a faixa fisica nao. */
    if (p < KNOB_RF_PCA_MIN) p = KNOB_RF_PCA_MIN;
    if (p > KNOB_RF_PCA_MAX) p = KNOB_RF_PCA_MAX;
    return p;
}
