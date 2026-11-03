#include "knob_senses_knn.h"

#include <math.h>
#include <stdlib.h>

/* O UNICO TU com o emlearn de vizinhos. ATENCAO dupla: as funcoes do eml_neighbors.h
 * sao globais (incluir em dois TUs duplicaria no LINK), e o proprio header NAO tem
 * include-guard (incluir duas vezes no MESMO TU redefine tudo). O knn_model.h gerado
 * pelo emlearn ja' faz o unico `#include <eml_neighbors.h>` - nao incluir por fora. */
#include "knn_model.h"

/* Media e desvio SO do split de treino (nunca validacao nem teste): as mesmas
 * constantes que o quantiza_z() do notebook usou para gerar os dados. */
static const float KNN_MEDIA[KNOB_KNN_N_FEATURES] = {
    27.570407f,
    527.957091f,
    41.595660f
};

static const float KNN_DESVIO[KNOB_KNN_N_FEATURES] = {
    2.699611f,
    138.248652f,
    6.110815f
};

/* Scratch da busca: 8 B por prototipo - E' o custo de RAM desta lib (o
 * `eml_neighbors_infer` exige o array de distancias completo). */
static EmlNeighborsDistanceItem s_distancias[KNOB_KNN_N_ITEMS];

/* Ordem TOTAL (distancia, depois indice). O quicksort do emlearn compara so' a
 * distancia - instavel nos empates; com um comparador total, QUALQUER algoritmo de
 * ordenacao da o mesmo resultado, e o emulador da verificacao reproduz. */
static int compara_distancia(const void *pa, const void *pb)
{
    const EmlNeighborsDistanceItem *a = (const EmlNeighborsDistanceItem *)pa;
    const EmlNeighborsDistanceItem *b = (const EmlNeighborsDistanceItem *)pb;
    if (a->distance != b->distance) {
        return (a->distance < b->distance) ? -1 : 1;
    }
    if (a->index != b->index) {
        return (a->index < b->index) ? -1 : 1;
    }
    return 0;
}

/* bruto -> z -> clamp -> int16. O clamp nao e' estetica: com ele a soma dos quadrados
 * das distancias fica < 2**30, que e' onde o `eml_isqrt` do emlearn travaria (o
 * `q <<= 2` estoura o int32 e o laco nunca termina). */
static int16_t quantiza_z(float z)
{
    if (z > KNOB_KNN_Z_CLAMP) {
        z = KNOB_KNN_Z_CLAMP;
    }
    if (z < -KNOB_KNN_Z_CLAMP) {
        z = -KNOB_KNN_Z_CLAMP;
    }
    return (int16_t)lrintf(z * KNOB_KNN_Z_SCALE);
}

float knn_pca_prever(const float bruto[KNOB_KNN_N_FEATURES])
{
    int16_t zq[KNOB_KNN_N_FEATURES];
    for (size_t i = 0; i < (size_t)KNOB_KNN_N_FEATURES; i++) {
        zq[i] = quantiza_z((bruto[i] - KNN_MEDIA[i]) / KNN_DESVIO[i]);
    }

    eml_neighbors_infer(&knn_modelo, zq, KNOB_KNN_N_FEATURES, s_distancias,
                        KNOB_KNN_N_ITEMS);
    qsort(s_distancias, KNOB_KNN_N_ITEMS, sizeof(EmlNeighborsDistanceItem),
          compara_distancia);

    /* A MEDIA dos k rotulos e' nossa: o emlearn 0.23.2 so' tem voto de classe (voto
     * discreto nao serve para regressao de kW). Inteira, com round-half-up: float nao
     * seria reproduzivel bit a bit pelo emulador da verificacao. */
    int32_t soma = 0;
    for (int i = 0; i < KNOB_KNN_K; i++) {
        soma += knn_modelo.labels[s_distancias[i].index];
    }
    const int32_t pred_mw = (2 * soma + KNOB_KNN_K) / (2 * KNOB_KNN_K);

    float kw = pred_mw / KNOB_KNN_Y_SCALE;
    if (kw < KNOB_KNN_PCA_MIN) {
        kw = KNOB_KNN_PCA_MIN;
    }
    if (kw > KNOB_KNN_PCA_MAX) {
        kw = KNOB_KNN_PCA_MAX;
    }
    return kw;
}
