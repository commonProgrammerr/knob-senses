"""Gera a biblioteca C do Random Forest a partir do joblib salvo pelo treino.

Uso:
    python utils/gerar_lib_rf.py 1h                  # gera a lib da granularidade 1h
    python utils/gerar_lib_rf.py 1h --medir          # gera e mede flash/RAM (xtensa)
    python utils/gerar_lib_rf.py 1h --sem-verificar  # gera sem o conferidor g++

O joblib (dist/random_forest_<freq>.joblib) e' escrito pelo
treino_random_forest.ipynb; este script NAO treina nada, apenas converte o
modelo salvo em codigo C (emlearn) e escreve a lib do PlatformIO
(lib/knob-senses-rf/). O firmware embarca uma lib por vez.
"""
import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "models" / "src"))

import joblib
import numpy as np

from common import contract, data, paths, verify
import rf


def carregar(freq: str):
    """O joblib salvo do treino (modelo, constantes, offset, metricas)."""
    assert freq in contract.GRANULARIDADES, f"freq={freq} fora de {contract.GRANULARIDADES}"
    caminho = paths.DIST / f"random_forest_{freq}.joblib"
    if not caminho.exists():
        raise FileNotFoundError(
            f"joblib nao encontrado: {caminho}\n"
            "Rode o treino_random_forest.ipynb antes (ele salva os joblibs).")
    return caminho, joblib.load(caminho)


def gerar_lib(freq: str) -> dict:
    """Carrega o joblib e escreve a lib C. Devolve {caminho: bytes}."""
    _, pkg = carregar(freq)
    return rf.generate_c_lib(
        pkg["modelo"], pkg["media"], pkg["desvio"],
        pkg["metricas"]["MAE (kW)"], pkg["metricas"]["R2"],
        offset=pkg["offset_calibracao_kw"], freq=freq)


def verificar(freq: str) -> None:
    """Conferidor g++: o C reproduz o modelo Python no teste dessa freq."""
    _, pkg = carregar(freq)
    conferidor = verify.write_checker()
    exe = verify.compile_host(conferidor)
    print(f"g++: conferidor compilado junto com o knob_senses_rf.cpp "
          f"({exe.stat().st_size / 1024:.1f} KB)")

    teste = data.load_split("test", freq=freq)
    bruto = teste[list(contract.RF_FEATURES)].to_numpy(np.float64)
    do_c = verify.predict(exe, bruto)

    media = np.array([pkg["media"][c] for c in contract.RF_FEATURES], dtype=np.float64)
    desvio = np.array([pkg["desvio"][c] for c in contract.RF_FEATURES], dtype=np.float64)
    X = ((bruto - media) / desvio).astype(np.float32)
    pred_py = pkg["modelo"].predict(X) + pkg["offset_calibracao_kw"]
    print(f"diferenca mediana C x Python: "
          f"{float(np.median(np.abs(do_c - pred_py))):.2e} kW")
    print("OK: a biblioteca reproduz a floresta treinada, com normalizacao, "
          "offset de calibracao e clip")


def medir() -> None:
    """Flash/RAM no toolchain xtensa do PlatformIO."""
    if not verify.XTENSA_GXX.exists():
        print("xtensa-g++ ausente: medicao de flash/RAM pulada.")
        return
    medida = verify.measure_xtensa()
    for nome, tamanho in medida["secoes"].items():
        print(f"  {nome:16} {tamanho:7} bytes")
    flash, ram = medida["flash"], medida["ram"]
    print(f"FLASH  {flash:7} bytes  ({flash / 1024:.1f} KB)")
    print(f"RAM    {ram:7} bytes")
    if ram:
        print(f"ATENCAO: {ram} bytes de RAM; esperado zero (emlearn emite codigo).")
    else:
        print("RAM zero: a floresta e' toda constante e vive no flash.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("freq", choices=contract.GRANULARIDADES,
                        help="granularidade embarcada (usa dist/random_forest_<freq>.joblib)")
    parser.add_argument("--verificar", action=argparse.BooleanOptionalAction, default=True,
                        help="roda o conferidor g++ apos gerar (default: sim)")
    parser.add_argument("--medir", action="store_true",
                        help="mede flash/RAM no toolchain xtensa")
    args = parser.parse_args(argv)

    escritos = gerar_lib(args.freq)
    for caminho in sorted(escritos):
        print(f"  {caminho.relative_to(RAIZ)!s:44} {escritos[caminho]:,} bytes")
    _, pkg = carregar(args.freq)
    print(f"offset embutido no C: {pkg['offset_calibracao_kw']:+.6f} kW (calibracao 2024H2)")
    print(f"granularidade embarcada: {args.freq} "
          f"({contract.GRANULARIDADE_LABEL[args.freq]})")

    if args.verificar:
        verificar(args.freq)
    if args.medir:
        medir()
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
