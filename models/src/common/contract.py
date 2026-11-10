"""O contrato do modelo: features, alvo e a ordem de entrada.

Este e' o unico lugar onde a ordem das features existe. Ela ja esteve copiada em
quatro lugares (notebook do RF, servidor de teste, header C e normalizacao.json) — e
uma dessas copias produziu o bug mais caro do projeto: dado ja' em z-score com o
modelo normalizando de novo, previsao colapsada num valor fixo e nenhum erro.
"""

# Ordem de entrada do Random Forest e dos eixos que o servidor transmite. Mudar aqui
# muda as duas pontas — e o assert do servidor pega se o normalizacao.json discordar.
RF_FEATURES = ("temp_amb", "ghit", "jacques")

TARGET = "PCA"         # potencia CA do inversor, em kW
INDEX = "datetime"

SEED = 42

# So' ifpee1 tem historico para treinar. O split temporal (train/calibration/test)
# vive no utils/dataset_prep.ipynb, que e' a fonte da verdade.
TRAIN_INVERSOR = "ifpee1"

# Granularidades do split (valores da coluna `freq` de data/splits/*.csv, escrita
# pelo utils/dataset_prep.ipynb), da mais fina para a mais grossa.
GRANULARIDADES = ("5min", "10min", "1h", "1D")

GRANULARIDADE_LABEL = {
    "5min": "5 min",
    "10min": "10 min",
    "1h": "1 h",
    "1D": "1 d",
}
