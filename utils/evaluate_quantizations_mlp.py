"""Compara quantizations do MLP no PC e recalibra o vies na calibracao.

Uso:
    python utils/evaluate_quantizations_mlp.py            # 1h, todas as variantes
    python utils/evaluate_quantizations_mlp.py 1h

As variantes saem do SavedModel do export e sao avaliadas no teste de 2025 com
o mesmo pre-processamento e pos-processamento do firmware. O offset de vies e'
recalculado SO na calibracao de 2024H2 e comparado com o offset atual do JSON.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.special import inv_boxcox
from scipy.stats import boxcox

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "models" / "src"))

from common import contract, data, gaps, metrics, paths  # noqa: E402
from mlp import transformacao  # noqa: E402


def load_variants(freq: str, amostras: np.ndarray) -> dict[str, bytes]:
    """Converte o SavedModel nas quantizations que o TFLite oferece."""
    saved = paths.EXPORT_MLP / freq / "saved_model"

    def representativo():
        for i in range(len(amostras)):
            yield [amostras[i:i + 1].astype(np.float32)]

    variantes = {}

    conversor = tf.lite.TFLiteConverter.from_saved_model(str(saved))
    variantes["float32"] = conversor.convert()

    conversor = tf.lite.TFLiteConverter.from_saved_model(str(saved))
    conversor.optimizations = [tf.lite.Optimize.DEFAULT]
    conversor.target_spec.supported_types = [tf.float16]
    variantes["float16"] = conversor.convert()

    conversor = tf.lite.TFLiteConverter.from_saved_model(str(saved))
    conversor.optimizations = [tf.lite.Optimize.DEFAULT]
    variantes["dynamic"] = conversor.convert()

    for nome in ("int8_io_int8", "int8_io_float"):
        conversor = tf.lite.TFLiteConverter.from_saved_model(str(saved))
        conversor.optimizations = [tf.lite.Optimize.DEFAULT]
        conversor.representative_dataset = representativo
        conversor.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        if nome == "int8_io_int8":
            conversor.inference_input_type = tf.int8
            conversor.inference_output_type = tf.int8
        else:
            conversor.inference_input_type = tf.float32
            conversor.inference_output_type = tf.float32
        variantes[nome] = conversor.convert()

    return variantes


def predict(conteudo: bytes, X: np.ndarray) -> np.ndarray:
    """Roda o .tflite no PC na mesma convencao de entrada do firmware."""
    it = tf.lite.Interpreter(model_content=conteudo)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    iscale, izp = inp["quantization"]
    oscale, ozp = out["quantization"]

    pred = np.empty(len(X), dtype=np.float64)
    for i, x in enumerate(X):
        if inp["dtype"] == np.int8:
            q = np.clip(np.rint(x / iscale) + izp, -128, 127).astype(np.int8)
            it.set_tensor(inp["index"], q.reshape(inp["shape"]))
        else:
            it.set_tensor(inp["index"], x.reshape(inp["shape"]).astype(inp["dtype"]))
        it.invoke()
        y = float(it.get_tensor(out["index"]).ravel()[0])
        pred[i] = (y - ozp) * oscale if out["dtype"] == np.int8 else y
    return pred


def to_kw(pred_bc: np.ndarray, lmbda: float, shift: float,
          clip: tuple[float, float]) -> np.ndarray:
    return inv_boxcox(np.clip(pred_bc, *clip), lmbda) - shift


def main(argv=None) -> int:
    freq = argv[0] if argv else "1h"
    dados = transformacao.carregar_e_validar(freq)
    minimo = np.array(dados["minimo"], dtype=np.float32)
    maximo = np.array(dados["maximo"], dtype=np.float32)
    shift = dados["shift_kw"]
    lmbda = dados["lambda"]
    clip = (dados["clip_min"], dados["clip_max"])
    offset_atual = dados["offset_kw"]
    pca_min, pca_max = dados["pca_min_kw"], dados["pca_max_kw"]

    # Teste de 2025, com os mesmos gaps do ensaio.
    caminho = paths.SPLITS / "test" / f"{contract.TRAIN_INVERSOR}.csv"
    teste = data.read_csv(caminho)
    teste = teste[teste["freq"] == freq].reset_index(drop=True)
    verdade = teste.copy()
    gapado, _, rotulo = gaps.gerar_gaps(teste, freq=freq)
    gapado["gap"] = rotulo
    X_tes = ((gapado[list(contract.RF_FEATURES)] - minimo) /
             (maximo - minimo)).to_numpy(np.float32)
    fora = gapado["PCA"].notna().to_numpy() & (gapado["PCA"].to_numpy() != 0.0)
    y_tes = gapado.loc[fora, "PCA"].to_numpy()

    # Calibracao de 2024H2, usada SO para o offset.
    cal = data.load_split("calibration", freq=freq)
    X_cal = ((cal[list(contract.RF_FEATURES)] - minimo) /
             (maximo - minimo)).to_numpy(np.float32)
    y_cal_kw = cal[contract.TARGET].to_numpy(np.float32)
    y_cal_bc = boxcox(y_cal_kw + shift, lmbda=lmbda)

    amostras = np.load(paths.EXPORT_MLP / freq / "dados_representativos.npy")
    variantes = load_variants(freq, amostras)
    destino = paths.EXPORT_MLP / freq / "quantizations"
    destino.mkdir(parents=True, exist_ok=True)

    linhas = []
    for nome, conteudo in variantes.items():
        (destino / f"{nome}.tflite").write_bytes(conteudo)

        pred_cal_bc = predict(conteudo, X_cal)
        pred_cal_kw = to_kw(pred_cal_bc, lmbda, shift, clip)
        offset = float(np.mean(y_cal_kw - pred_cal_kw))

        pred_tes_bc = predict(conteudo, X_tes[fora])
        base = to_kw(pred_tes_bc, lmbda, shift, clip)
        pred_atual = np.clip(base + offset_atual, pca_min, pca_max)
        pred_novo = np.clip(base + offset, pca_min, pca_max)

        r_atual = metrics.evaluate(f"{nome} offset atual", y_tes, pred_atual,
                                   pca_min, pca_max)
        r_novo = metrics.evaluate(f"{nome} offset recalibrado", y_tes, pred_novo,
                                  pca_min, pca_max)

        linhas.append({
            "variante": nome,
            "KB": len(conteudo) / 1024,
            "offset atual": offset_atual,
            "offset novo": offset,
            "MAE atual": r_atual["MAE (kW)"],
            "MAE novo": r_novo["MAE (kW)"],
            "R2 atual": r_atual["R2"],
            "R2 novo": r_novo["R2"],
            "vies atual": r_atual["vies (kW)"],
            "vies novo": r_novo["vies (kW)"],
        })

    tabela = pd.DataFrame(linhas)
    print()
    print(tabela.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
