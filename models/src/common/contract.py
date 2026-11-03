"""O contrato do modelo: features, alvo e a ordem do split.

Este e' o unico lugar onde a ordem das features existe. Ela ja esteve copiada em
quatro lugares (notebook do RF, servidor de teste, header C e normalizacao.json) — e
uma dessas copias produziu o bug mais caro do projeto: dado ja' em z-score com o
modelo normalizando de novo, previsao colapsada num valor fixo e nenhum erro.

Os dois tuples de features sao DIFERENTES de proposito:

- RF_FEATURES    — a ordem de entrada do modelo e dos eixos do servidor;
- SPLIT_FEATURES — a ordem das colunas de feature no CSV do split.

O split nao tem lag: escreve INDEX, TARGET e SPLIT_FEATURES.
"""

# Ordem de entrada do Random Forest e dos eixos que o servidor transmite. Mudar aqui
# muda as duas pontas — e o assert do servidor pega se o normalizacao.json discordar.
# RF_FEATURES = ("ghit", "jacques", "temperatura_ambiente")
RF_FEATURES = ("temperatura_ambiente", "ghit", "jacques")

# As colunas de feature que o split escreve (sem o alvo).
SPLIT_FEATURES = ("ghit", "jacques", "temperatura_ambiente")

TARGET = "PCA"         # potencia CA do inversor, em kW
INDEX = "datetime"

SEED = 42

# O split manual: so' a estacao ifpee1 tem historico para treinar; o corte e' por ano,
# sem embaralhar (validacao e teste sao futuro de verdade). Anos inclusivos, None = sem
# limite. "train" = anos < 2023 e' equivalente a (None, 2022) porque o dado comeca em 2021.
TRAIN_INVERSOR = "ifpee1"
SPLIT_CUTS = {"train": (None, 2022), "validation": (2023, 2023), "test": (2024, 2025)}

SPLIT_COLUMNS = (INDEX, TARGET, *SPLIT_FEATURES)
