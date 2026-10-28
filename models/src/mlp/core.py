"""O caminho do MLP (Keras): preparo dos dados, rede, treino e export.

Adaptado do antigo treino_regressao.ipynb ao contrato atual: as 4 features do contrato
mais o PCA da hora anterior, alvo PCA, no mesmo split por ano. O TensorFlow e' importado
DENTRO das funcoes - `import mlp` nao exige TF, entao o pacote instala no .venv de dados.
"""

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from common import contract, data, paths

# As entradas do MLP: o contrato do RF/k-NN mais o lag que o split ja' escreve.
ENTRADAS = (*contract.RF_FEATURES, contract.LAG)
OCULTAS = (64, 32)
EPOCHS = 200
BATCH = 512
PACIENCIA = 25


def _tf():
    """Importa TensorFlow/Keras sob demanda, com erro que diz o que fazer."""
    try:
        import keras
        import tensorflow as tf
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "o MLP precisa de TensorFlow/Keras: instale o extra `mlp` com "
            "`pip install -e \".[mlp]\"` num ambiente com Python <= 3.13 "
            "(o .venv de dados e' 3.14)."
        ) from e
    return tf, keras


def prepara():
    """Splits atuais em z-score -> arrays float32 e as constantes de normalizacao.

    A normalizacao e' ajustada SO no treino; validacao e teste usam as mesmas
    constantes. Devolve `(dados, media, desvio)`, onde `dados` tem as chaves
    treino/validacao/teste, cada uma com X e y.
    """
    frames = {nome: data.load_split(nome) for nome in ("train", "validation", "test")}
    media, desvio = data.fit_normalization(frames["train"], ENTRADAS)

    dados = {}
    for arquivo, chave in (("train", "treino"), ("validation", "validacao"),
                           ("test", "teste")):
        d = frames[arquivo]
        dados[chave] = {
            "X": data.z_score(d, media, desvio, ENTRADAS).to_numpy(np.float32),
            "y": d[contract.TARGET].to_numpy(np.float32),
        }
    return dados, media, desvio


def build(ocultas=OCULTAS, n_entradas=len(ENTRADAS)):
    """A rede 5 -> 64 -> 32 -> 1: densa, ReLU, saida linear em kW."""
    _, keras = _tf()
    keras.utils.set_random_seed(contract.SEED)
    camadas = [keras.layers.Input(shape=(n_entradas,), name="features")]
    for i, n in enumerate(ocultas):
        camadas.append(keras.layers.Dense(n, activation="relu", name=f"oculta_{i}"))
    camadas.append(keras.layers.Dense(1, name="pca_kw"))

    modelo = keras.Sequential(camadas, name="knob_senses_pca")
    modelo.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return modelo


def train(modelo, dados, epochs=EPOCHS, batch_size=BATCH, paciencia=PACIENCIA):
    """Treina com early stopping no val_loss e devolve o historico."""
    _, keras = _tf()
    chamadas = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=paciencia, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5, verbose=1),
    ]
    return modelo.fit(
        dados["treino"]["X"], dados["treino"]["y"],
        validation_data=(dados["validacao"]["X"], dados["validacao"]["y"]),
        epochs=epochs, batch_size=batch_size, callbacks=chamadas, verbose=0)


def export(modelo, dados, media, desvio, destino=paths.EXPORT_MLP):
    """Escreve SavedModel (batch 1), zip do Edge Impulse, .tflite, .npy e JSON.

    Devolve `{caminho: tamanho em bytes}`. O `.npy` de quantizacao usa a VALIDACAO - o teste nao
    entra aqui, senao a quantizacao veria o futuro.
    """
    tf, _ = _tf()
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)

    assinatura = [tf.TensorSpec(shape=(1, len(ENTRADAS)), dtype=tf.float32,
                                name="features")]
    dir_saved = destino / "saved_model"
    if dir_saved.exists():
        shutil.rmtree(dir_saved)
    modelo.export(dir_saved, input_signature=assinatura, verbose=False)

    caminho_zip = destino / "modelo_edge_impulse.zip"
    with zipfile.ZipFile(caminho_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for arquivo in sorted(dir_saved.rglob("*")):
            if arquivo.is_file():
                z.write(arquivo, arquivo.relative_to(destino))

    conversor = tf.lite.TFLiteConverter.from_saved_model(str(dir_saved))
    caminho_tflite = destino / "modelo.tflite"
    caminho_tflite.write_bytes(conversor.convert())

    amostra = dados["validacao"]["X"][:2000].reshape(-1, 1, len(ENTRADAS))
    caminho_npy = destino / "dados_representativos.npy"
    np.save(caminho_npy, amostra.astype(np.float32))

    caminho_json = destino / "normalizacao.json"
    caminho_json.write_text(json.dumps({
        "modelo": "knob-senses-mlp",
        "gerado_por": "models/src/mlp",
        "janela_horas": 1,
        "lag": contract.LAG,
        "ordem_das_entradas": list(ENTRADAS),
        "features": list(ENTRADAS),
        "features_base": list(contract.RF_FEATURES),
        "media": {c: float(media[c]) for c in ENTRADAS},
        "desvio": {c: float(desvio[c]) for c in ENTRADAS},
        "formula": "x_normalizado = (x_bruto - media) / desvio",
        "fonte": "models/src/mlp (ajustado somente no split de treino)",
        "alvo": {
            "nome": contract.TARGET,
            "unidade": "kW CA",
            "horizonte": "t (mesma hora)",
            "min": contract.PCA_MIN,
            "max": contract.PCA_MAX,
            "observacao": "a saida do modelo ja esta em kW; nao ha transformacao inversa",
        },
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {p: p.stat().st_size for p in
            (caminho_zip, caminho_tflite, caminho_npy, caminho_json)}
