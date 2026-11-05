"""O caminho do MLP (Keras): export do modelo treinado.

A definicao da rede, o preparo dos splits e o treino vivem em
`utils/treino_mlp.ipynb`; este modulo so' exporta o modelo. O TensorFlow e'
importado DENTRO do export - `import mlp` nao exige TF.
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


def export(modelo, amostras, minimo, maximo, destino=paths.EXPORT_MLP, offset: float = 0.0,
           target_transform: dict | None = None):
    """Escreve SavedModel (batch 1), zip do Edge Impulse, .tflite, .npy e JSON.

    `amostras` sao entradas representativas JA' NA ESCALA DO MODELO (ex.: X da
    calibracao apos o min-max) para a quantizacao - o teste nao entra aqui, senao a
    quantizacao veria o futuro.
    `offset` e' o vies em kW medido na calibracao 2024H2, aplicado DEPOIS da
    inversa. `target_transform` registra a transformacao do alvo (ex.: Box-Cox com
    lambda/clip); se ela existir, a saida do modelo NAO esta em kW e o
    pos-processamento precisa desfazer.
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

    amostra = np.asarray(amostras, dtype=np.float32)[:2000].reshape(-1, 1, len(ENTRADAS))
    caminho_npy = destino / "dados_representativos.npy"
    np.save(caminho_npy, amostra)

    pca_min, pca_max = data.physical_target_limits()
    payload = {
        "modelo": "knob-senses-mlp",
        "gerado_por": "models/src/mlp",
        "janela_horas": 1,
        "normalizacao": "min_max",
        "ordem_das_entradas": list(ENTRADAS),
        "features": list(ENTRADAS),
        "minimo": {c: float(minimo[c]) for c in ENTRADAS},
        "maximo": {c: float(maximo[c]) for c in ENTRADAS},
        "formula": "x_normalizado = (x_bruto - minimo) / (maximo - minimo)",
        "fonte": "models/src/mlp (treino 2021..2024H1)",
        "calibracao_offset_kw": float(offset),
        "calibracao_ajustada_em": "2024H2 (nunca no teste)",
        "alvo": {
            "nome": contract.TARGET,
            "unidade": "kW CA",
            "horizonte": "t (mesma hora)",
            "min": pca_min,
            "max": pca_max,
            "observacao": "a saida do modelo ja esta em kW; nao ha transformacao inversa",
        },
    }
    if target_transform is not None:
        payload["transformacao_alvo"] = target_transform
        payload["alvo"]["observacao"] = (
            "a saida do modelo esta no espaco transformado; aplique "
            "transformacao_alvo (clip_boxcox, inv_boxcox, offset_kw e clip fisico) "
            "para obter kW")
    caminho_json = destino / "normalizacao.json"
    caminho_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {p: p.stat().st_size for p in
            (caminho_zip, caminho_tflite, caminho_npy, caminho_json)}
