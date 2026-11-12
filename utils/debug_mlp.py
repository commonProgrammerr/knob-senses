"""Dumpa as primeiras linhas do caminho int8 no PC para comparar com a placa.

Uso:
    python utils/debug_mlp.py          # 8 linhas, 1h, modelo_quantizado
    python utils/debug_mlp.py 1h 16

A saida tem os mesmos campos do debug do firmware: bruto, normalizado, q de
entrada, Box-Cox, q de saida e kW.
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.special import inv_boxcox

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "models" / "src"))

from common import contract, data, gaps, paths  # noqa: E402
from mlp import transformacao  # noqa: E402


def main(argv=None) -> int:
    freq = argv[0] if argv else "1h"
    n = int(argv[1]) if len(argv) > 1 else 8
    d = transformacao.carregar_e_validar(freq)
    minimo = np.array(d["minimo"], dtype=np.float32)
    maximo = np.array(d["maximo"], dtype=np.float32)

    modelo = paths.EXPORT_MLP / freq / "modelo_quantizado.tflite"
    it = tf.lite.Interpreter(model_path=str(modelo))
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    iscale, izp = inp["quantization"]
    oscale, ozp = out["quantization"]

    teste = data.read_csv(paths.SPLITS / "test" / f"{contract.TRAIN_INVERSOR}.csv")
    teste = teste[teste["freq"] == freq].reset_index(drop=True)
    gapado, _, _ = gaps.gerar_gaps(teste, freq=freq)
    X = ((gapado[list(contract.RF_FEATURES)] - minimo) /
         (maximo - minimo)).to_numpy(np.float32)

    for i in range(min(n, len(X))):
        bruto = gapado.iloc[i][list(contract.RF_FEATURES)].to_numpy(np.float32)
        norm = X[i]
        q = np.clip(np.rint(norm / iscale) + izp, -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], q.reshape(inp["shape"]))
        it.invoke()
        y = float(it.get_tensor(out["index"]).ravel()[0])
        bc = (y - ozp) * oscale
        qout = int(round(bc / oscale) + ozp)
        kw = inv_boxcox(np.clip(bc, d["clip_min"], d["clip_max"]), d["lambda"])
        kw = float(np.clip(kw - d["shift_kw"] + d["offset_kw"],
                           d["pca_min_kw"], d["pca_max_kw"]))
        print(
            f"dbg {i} | bruto {bruto[0]:.4f} {bruto[1]:.4f} {bruto[2]:.4f} | "
            f"norm {norm[0]:.5f} {norm[1]:.5f} {norm[2]:.5f} | "
            f"q {q[0]} {q[1]} {q[2]} | bc {bc:.6f} | qout {qout} | kw {kw:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
