"""O caminho do MLP (Keras): preparo dos dados e export.

A definicao da rede e o treino vivem em `utils/treino_regressao.ipynb`; este modulo
so' monta os arrays de treino/validacao/teste e exporta o modelo treinado. O
TensorFlow e' importado DENTRO do export - `import mlp` nao exige TF.
"""

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from common import contract, data, paths

# As entradas do MLP sao as features do contrato.
ENTRADAS = contract.RF_FEATURES


def _tf():
    """Importa TensorFlow/Keras sob demanda, com erro que diz o que fazer."""
    try:
        import keras
        import tensorflow as tf
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "o MLP precisa de TensorFlow/Keras: instale as dependencias do projeto "
            "com `pip install -e .` no ambiente do projeto."
        ) from e
    return tf, keras


def prepara():
    """Splits atuais em min-max -> arrays float32 e as constantes de normalizacao.

    O min/max e' ajustado SO no treino; validacao e teste usam as mesmas constantes.
    Devolve `(dados, minimo, maximo)`, onde `dados` tem as chaves
    treino/validacao/teste, cada uma com X e y.
    """
    frames = {nome: data.load_split(nome) for nome in ("train", "validation", "test")}
    minimo, maximo = data.fit_min_max(frames["train"], ENTRADAS)

    dados = {}
    for arquivo, chave in (("train", "treino"), ("validation", "validacao"),
                           ("test", "teste")):
        d = frames[arquivo]
        dados[chave] = {
            "X": data.min_max_scale(d, minimo, maximo, ENTRADAS).to_numpy(np.float32),
            "y": d[contract.TARGET].to_numpy(np.float32),
        }
    return dados, minimo, maximo


def export(modelo, dados, minimo, maximo, destino=paths.EXPORT_MLP):
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

    pca_min, pca_max = data.target_limits()
    caminho_json = destino / "normalizacao.json"
    caminho_json.write_text(json.dumps({
        "modelo": "knob-senses-mlp",
        "gerado_por": "models/src/mlp",
        "janela_horas": 1,
        "normalizacao": "min_max",
        "ordem_das_entradas": list(ENTRADAS),
        "features": list(ENTRADAS),
        "minimo": {c: float(minimo[c]) for c in ENTRADAS},
        "maximo": {c: float(maximo[c]) for c in ENTRADAS},
        "formula": "x_normalizado = (x_bruto - minimo) / (maximo - minimo)",
        "fonte": "models/src/mlp (ajustado somente no split de treino)",
        "alvo": {
            "nome": contract.TARGET,
            "unidade": "kW CA",
            "horizonte": "t (mesma hora)",
            "min": pca_min,
            "max": pca_max,
            "observacao": "a saida do modelo ja esta em kW; nao ha transformacao inversa",
        },
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {p: p.stat().st_size for p in
            (caminho_zip, caminho_tflite, caminho_npy, caminho_json)}
