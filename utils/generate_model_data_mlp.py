"""Gera o model_data.cc da lib/knob-mlp-tflite a partir do .tflite escolhido.

Uso:
    python utils/generate_model_data_mlp.py 1h int8
    python utils/generate_model_data_mlp.py 1h float32
    python utils/generate_model_data_mlp.py 1h dynamic
    python utils/generate_model_data_mlp.py 1h float16

O int8 e o float32 saem dos artefatos principais do export; dynamic e float16
saem de dist/{freq}/quantizations/, gerados por utils/evaluate_quantizations_mlp.py.
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "models" / "src"))

from common import paths  # noqa: E402

PRIMARY = {
    "int8": "modelo_quantizado.tflite",
    "float32": "modelo.tflite",
}


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    if len(argv) < 2:
        print("uso: generate_model_data_mlp.py <freq> <quantization>")
        return 2
    freq, quantization = argv[0], argv[1]
    base = paths.EXPORT_MLP / freq
    if quantization in PRIMARY:
        origem = base / PRIMARY[quantization]
    else:
        origem = base / "quantizations" / f"{quantization}.tflite"
    if not origem.exists():
        print(f"artefato ausente: {origem}")
        return 1

    data = origem.read_bytes()
    linhas = [
        '#include "knob_mlp_model.h"',
        "",
        "extern const unsigned char g_knob_mlp_model[] = {",
    ]
    for i in range(0, len(data), 12):
        linhas.append("  " + ", ".join(f"0x{b:02x}" for b in data[i:i+12]) + ",")
    linhas.append("};")
    linhas.append(f"extern const unsigned int g_knob_mlp_model_len = {len(data)};")
    destino = RAIZ / "lib" / "knob-mlp-tflite" / "src" / "model_data.cc"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"model_data.cc: {quantization} ({len(data)} bytes) em {destino.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
