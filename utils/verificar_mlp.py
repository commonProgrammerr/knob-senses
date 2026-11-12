"""Verifica o pos-processamento do MLP e simula o caminho int8 sem a placa.

Uso:
    python utils/verificar_mlp.py 1h            # check + host + simulacao
    python utils/verificar_mlp.py 1h --check    # header e modelo embutido
    python utils/verificar_mlp.py 1h --host     # funcoes C++ contra Python
    python utils/verificar_mlp.py 1h --simular  # modelo int8 no teste de 2025
    python utils/verificar_mlp.py 1h --simular --modelo modelo.tflite  # float32

O --host compila include/mlp_posprocessamento.h com g++ e compara com a conta
em Python. O --simular abre dist/{freq}/modelo_quantizado.tflite (ou o arquivo
passado em --modelo), normaliza as features, quantiza a entrada quando o modelo
e' int8, desquantiza a saida e aplica a mesma sequencia do firmware; e' o numero
que o ensaio na placa deve reproduzir.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "models" / "src"))

from common import contract, data, gaps, metrics, paths  # noqa: E402
from mlp import transformacao  # noqa: E402

_HARNESS = r"""
#include <stdio.h>
#include "mlp_posprocessamento.h"

int main(int argc, char **argv)
{
  if (argc > 1 && argv[1][0] == 'z')
  {
    float z;
    while (scanf("%f", &z) == 1) printf("%.9g\n", mlp_posprocessar_saida(z));
    return 0;
  }
  float bruto[3], norm[3];
  while (scanf("%f %f %f", &bruto[0], &bruto[1], &bruto[2]) == 3)
  {
    mlp_normalizar(bruto, norm);
    printf("%.9g %.9g %.9g\n", norm[0], norm[1], norm[2]);
  }
  return 0;
}
"""


def _check(freq: str, modelo_nome: str = "modelo_quantizado.tflite") -> bool:
    ok, mensagem = transformacao.verificar_sincronia(freq)
    print(mensagem)
    if not ok:
        return False
    mensagem = transformacao.verificar_modelo_embutido(freq, modelo_nome)
    if mensagem is None:
        print(f"ERRO: {paths.EXPORT_MLP / freq / modelo_nome} ausente")
        return False
    print(mensagem)
    return True


def _rodar(exe: Path, linhas: list[str], modo: str | None = None) -> list[str]:
    cmd = [str(exe)] + ([modo] if modo else [])
    resultado = subprocess.run(
        cmd, input="\n".join(linhas) + "\n", text=True,
        capture_output=True, check=True)
    return resultado.stdout.strip().splitlines()


def _host(freq: str) -> bool:
    dados = transformacao.carregar_e_validar(freq)
    compilador = shutil.which("g++")
    if compilador is None:
        print("ERRO: g++ nao encontrado no PATH")
        return False

    minimo = np.array(dados["minimo"], dtype=np.float64)
    maximo = np.array(dados["maximo"], dtype=np.float64)
    meio = (minimo + maximo) / 2.0
    triplas = np.vstack([minimo, maximo, meio, minimo - 1.0, maximo + 1.0])
    esperado_norm = (triplas - minimo) / (maximo - minimo)

    zs = np.array([
        0.0,
        dados["clip_min"],
        dados["clip_max"],
        (dados["clip_min"] + dados["clip_max"]) / 2.0,
        dados["clip_min"] - 1.0,
        dados["clip_max"] + 1.0,
        -1.3753749,
        0.872834,
    ])
    from scipy.special import inv_boxcox

    z_clip = np.clip(zs, dados["clip_min"], dados["clip_max"])
    esperado_kw = inv_boxcox(z_clip, dados["lambda"])
    esperado_kw = esperado_kw - dados["shift_kw"] + dados["offset_kw"]
    esperado_kw = np.clip(
        esperado_kw, dados["pca_min_kw"], dados["pca_max_kw"])

    with tempfile.TemporaryDirectory() as tmp:
        fonte = Path(tmp) / "harness.cpp"
        fonte.write_text(_HARNESS, encoding="utf-8")
        exe = Path(tmp) / ("harness.exe" if os.name == "nt" else "harness")
        subprocess.run([
            compilador, "-O2", "-std=gnu++11",
            "-I", str(RAIZ / "include"),
            "-I", str(paths.LIB_MLP_SRC),
            str(fonte), "-lm", "-o", str(exe),
        ], check=True)

        linhas = [" ".join(f"{v:.9g}" for v in linha) for linha in triplas]
        obtido_norm = np.array([
            [float(v) for v in linha.split()]
            for linha in _rodar(exe, linhas)
        ])
        erro_norm = float(np.max(np.abs(obtido_norm - esperado_norm)))

        obtido_kw = np.array([
            float(linha) for linha in _rodar(
                exe, [f"{z:.9g}" for z in zs], modo="z")
        ])
        erro_kw = float(np.max(np.abs(obtido_kw - esperado_kw)))

    print(f"normalizacao: erro maximo {erro_norm:.3e}")
    print(f"inversa:      erro maximo {erro_kw:.3e} kW")
    if erro_norm > 1e-6:
        print("ERRO: normalizacao fora da tolerancia (1e-6)")
        return False
    if erro_kw > 1e-4:
        print("ERRO: inversa fora da tolerancia (1e-4 kW)")
        return False
    print("OK: as funcoes do firmware reproduzem a conta em Python")
    return True


def _simular(freq: str, modelo_nome: str = "modelo_quantizado.tflite") -> bool:
    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        print("ERRO: TensorFlow nao instalado; --simular pulado")
        return False
    from scipy.special import inv_boxcox

    dados = transformacao.carregar_e_validar(freq)
    caminho = paths.SPLITS / "test" / f"{contract.TRAIN_INVERSOR}.csv"
    if not caminho.exists():
        print(f"ERRO: {caminho} nao existe")
        return False
    teste = data.read_csv(caminho)
    teste = teste[teste["freq"] == freq].reset_index(drop=True)
    verdade = teste.copy()
    gapado, _, rotulo = gaps.gerar_gaps(teste, freq=freq)
    gapado["gap"] = rotulo

    X = gapado[list(contract.RF_FEATURES)].to_numpy(np.float32)
    minimo = np.array(dados["minimo"], dtype=np.float32)
    maximo = np.array(dados["maximo"], dtype=np.float32)
    X = ((X - minimo) / (maximo - minimo)).astype(np.float32)

    modelo = Path(modelo_nome)
    if not modelo.is_absolute():
        modelo = paths.EXPORT_MLP / freq / modelo_nome
    if not modelo.exists():
        print(f"ERRO: {modelo} nao existe")
        return False
    it = tf.lite.Interpreter(model_path=str(modelo))
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    iscale, izp = inp["quantization"]
    oscale, ozp = out["quantization"]
    nome_modelo = "int8" if inp["dtype"] == np.int8 else "float32"
    print(f"modelo: {modelo}")

    pred_bc = np.empty(len(X), dtype=np.float64)
    for i, x in enumerate(X):
        if inp["dtype"] == np.int8:
            q = np.clip(np.rint(x / iscale) + izp, -128, 127).astype(np.int8)
            it.set_tensor(inp["index"], q.reshape(inp["shape"]))
        else:
            it.set_tensor(inp["index"], x.reshape(inp["shape"]).astype(inp["dtype"]))
        it.invoke()
        y = float(it.get_tensor(out["index"]).ravel()[0])
        pred_bc[i] = (y - ozp) * oscale if out["dtype"] == np.int8 else y

    z = np.clip(pred_bc, dados["clip_min"], dados["clip_max"])
    pred = inv_boxcox(z, dados["lambda"])
    pred = pred - dados["shift_kw"] + dados["offset_kw"]
    pred = np.clip(pred, dados["pca_min_kw"], dados["pca_max_kw"])

    def _mascara(real):
        return real.notna().to_numpy() & (real.to_numpy() != 0.0)

    fora = _mascara(gapado["PCA"])
    dentro = gapado["PCA"].isna().to_numpy()
    real_dentro = verdade.loc[dentro, "PCA"]
    ok_dentro = real_dentro.notna().to_numpy() & (real_dentro.to_numpy() != 0.0)
    completo = _mascara(verdade["PCA"])

    metrics.evaluate(
        f"{nome_modelo} {freq} fora dos gaps",
        gapado.loc[fora, "PCA"].to_numpy(),
        pred[fora], dados["pca_min_kw"], dados["pca_max_kw"])
    metrics.evaluate(
        f"{nome_modelo} {freq} dentro dos gaps",
        real_dentro.to_numpy()[ok_dentro],
        pred[dentro][ok_dentro], dados["pca_min_kw"], dados["pca_max_kw"])
    metrics.evaluate(
        f"{nome_modelo} {freq} teste completo",
        verdade.loc[completo, "PCA"].to_numpy(),
        pred[completo], dados["pca_min_kw"], dados["pca_max_kw"])

    print(f"pred: min {pred.min():.4f} kW | max {pred.max():.4f} kW | "
          f"{np.unique(pred).size} valores distintos | "
          f"{int(np.sum(pred >= 1.99))} em >= 1.99 kW")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("freq", nargs="?", default="1h",
                        choices=contract.GRANULARIDADES)
    parser.add_argument("--check", action="store_true",
                        help="header e modelo embutido")
    parser.add_argument("--host", action="store_true",
                        help="funcoes C++ contra Python")
    parser.add_argument("--simular", action="store_true",
                        help="modelo no teste de 2025")
    parser.add_argument("--modelo", default="modelo_quantizado.tflite",
                        help="arquivo do modelo em dist/{freq}/ "
                             "(default: modelo_quantizado.tflite)")
    args = parser.parse_args(argv)

    if not (args.check or args.host or args.simular):
        args.check = args.host = args.simular = True

    ok = True
    if args.check:
        ok = _check(args.freq, args.modelo) and ok
    if args.host:
        ok = _host(args.freq) and ok
    if args.simular:
        ok = _simular(args.freq, args.modelo) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
