"""O contrato do modelo: features, alvo, faixa fisica e cortes do split.

Este e' o unico lugar onde a ordem das features existe. Ela ja esteve copiada em
quatro lugares (notebook do RF, servidor de teste, header C e normalizacao.json) — e
uma dessas copias produziu o bug mais caro do projeto: dado ja' em z-score com o
modelo normalizando de novo, previsao colapsada num valor fixo e nenhum erro.

Os dois tuples de features sao DIFERENTES de proposito:

- RF_FEATURES    — a ordem de entrada do modelo e dos eixos do servidor;
- SPLIT_FEATURES — a ordem das colunas de feature no CSV do split.

O PCA_lag1 (LAG) e' escrito no split e usado pelo MLP; o RF e o k-NN treinam sem ele.
"""

# Ordem de entrada do Random Forest e dos eixos que o servidor transmite. Mudar aqui
# muda as duas pontas — e o assert do servidor pega se o normalizacao.json discordar.
# RF_FEATURES = ("ghit", "jacques", "temperatura_ambiente")
RF_FEATURES = ("temperatura_ambiente", "ghit", "jacques", "sintetico")

# As colunas de feature que o split escreve (sem o alvo e sem o lag).
SPLIT_FEATURES = ("ghit", "jacques", "temperatura_ambiente", "sintetico")

TARGET = "PCA"         # potencia CA do inversor, em kW
INDEX = "datetime"
LAG = "PCA_lag1"       # PCA da hora anterior, como lag de tempo real

SEED = 42

# Faixa fisica do alvo: o inversor satura em ~2 kW e o dataset nao tem zero (o 0 era
# sentinela de faltante, removido na analise). Prever fora desta faixa e' impossivel.
PCA_MIN, PCA_MAX = 0.01, 2.02

# Limites de sanidade: os CSV tem que estar em GRANDEZA FISICA. Se alguem voltar a
# normalizar no prep, estes limites quebram no treino — e nao em campo, com o firmware
# normalizando duas vezes e a previsao colapsando num valor fixo.
LIMITS = {
    "temperatura_ambiente": (-20.0, 60.0),
    "ghit": (0.0, 1500.0),
    "jacques": (-20.0, 90.0),
    "sintetico": (0.0, 1.0),
}

# O split manual: so' a estacao ifpee1 tem historico para treinar; o corte e' por ano,
# sem embaralhar (validacao e teste sao futuro de verdade). Anos inclusivos, None = sem
# limite. "train" = anos < 2023 e' equivalente a (None, 2022) porque o dado comeca em 2021.
TRAIN_STATION = "ifpee1"
SPLIT_CUTS = {"train": (None, 2022), "validation": (2023, 2023), "test": (2024, 2024)}

SPLIT_COLUMNS = (INDEX, TARGET, LAG, *SPLIT_FEATURES)
