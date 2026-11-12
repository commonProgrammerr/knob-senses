"""Gera o header C++ com as constantes do MLP a partir dos JSONs do treino.

Uso:
    python utils/gerar_header_mlp.py 1h            # valida e escreve o header
    python utils/gerar_header_mlp.py 1h --check    # nao escreve; falha se dessincronizado

O header gerado é include/mlp_transformacao_alvo.h. Ele contem o
min-max das entradas, o shift, o lambda, o clip de Box-Cox, o offset de
calibracao e a faixa fisica do alvo. O firmware inclui esse header em
include/mlp_posprocessamento.h.
"""
import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "models" / "src"))

from common import contract  # noqa: E402
from mlp import transformacao  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("freq", choices=contract.GRANULARIDADES,
                        help="granularidade do modelo (dist/{freq}/)")
    parser.add_argument("--check", action="store_true",
                        help="nao escreve; falha se o header estiver dessincronizado")
    args = parser.parse_args(argv)

    if args.check:
        ok, mensagem = transformacao.verificar_sincronia(args.freq)
        print(mensagem)
        return 0 if ok else 1

    caminho = transformacao.gerar_header(args.freq)
    print(f"header gerado: {caminho.relative_to(RAIZ)}")
    print(f"  granularidade: {args.freq}")
    print(f"  features: {', '.join(contract.RF_FEATURES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
