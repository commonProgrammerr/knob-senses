"""A prova de que o C gerado reproduz a floresta do sklearn.

O g++ do host confere a MATEMATICA compilando o proprio knob_senses_rf.cpp da entrega
(nunca uma copia); o xtensa mede o FLASH no alvo de verdade. O que o teste nao pode
exigir e' igualdade exata: o sklearn avalia em float64, o C em float32 — e float32 e' o
que o ESP32 tem. O esperado e' quase tudo no ruido, com um punhado de trocas de folha.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import contract, paths

GXX = shutil.which("g++") or "g++"
XTENSA_BIN = Path.home() / ".platformio" / "packages" / "toolchain-xtensa-esp32s3" / "bin"
XTENSA_GXX = XTENSA_BIN / "xtensa-esp32s3-elf-g++.exe"
XTENSA_SIZE = XTENSA_BIN / "xtensa-esp32s3-elf-size.exe"

# Uma linha "trocou de folha" quando a diferenca e' grande demais para ser ruido de
# float32. 1e-4 kW esta' duas ordens acima do epsilon e tres abaixo do MAE do modelo.
TOLERANCE = 1e-4
MAX_SWAPS = 0.005


def write_checker(dest: Path = paths.SCRATCH / "verifica.cpp", *,
                  header: str = "knob_senses_rf.h", func: str = "pca_prever",
                  n_macro: str = "KNOB_RF_N_FEATURES",
                  n_features: int = len(contract.RF_FEATURES)) -> Path:
    """O main em C++ que le' valores BRUTOS da entrada padrao e imprime func(x).

    Os parametros nomeados existem para o k-NN reusar este conferidor; os defaults sao
    o caminho do RF - o notebook do RF chama sem argumento e nao muda.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = n_features
    scan_fmt = " ".join("%f" for _ in range(n))
    arguments = ", ".join(f"&x[{i}]" for i in range(n))
    dest.write_text(
        '#include <cstdio>\n'
        f'#include "{header}"\n'
        '\n'
        'int main()\n'
        '{\n'
        f'    float x[{n_macro}];\n'
        f'    while (std::scanf("{scan_fmt}", {arguments})\n'
        f'           == {n_macro}) {{\n'
        f'        std::printf("%.8f\\n", {func}(x));\n'
        '    }\n'
        '    return 0;\n'
        '}\n', encoding="utf-8")
    return dest


def compile_host(checker: Path = paths.SCRATCH / "verifica.cpp",
                 exe: Path = paths.SCRATCH / "verifica.exe", *,
                 lib_src: Path = paths.LIB_RF_SRC,
                 impl: str = "knob_senses_rf.cpp") -> Path:
    """Compila o conferidor JUNTO com a implementacao da lib (o arquivo da entrega)."""
    exe = Path(exe)
    r = subprocess.run(
        f'"{GXX}" -O2 -std=c++17 -o "{exe}" "{checker}" '
        f'"{Path(lib_src) / impl}" -I"{Path(lib_src)}" -lm',
        shell=True, capture_output=True, text=True)
    assert r.returncode == 0, f"g++ falhou:\n{r.stderr}"
    return exe


def predict(exe: Path, matrix) -> np.ndarray:
    """Empurra as linhas da matriz (brutas) pelo conferidor e devolve as previsoes."""
    payload = "\n".join(" ".join(f"{v:.9g}" for v in row) for row in matrix)
    r = subprocess.run([str(exe)], input=payload, capture_output=True, text=True)
    assert r.returncode == 0, f"o conferidor falhou:\n{r.stderr}"
    predictions = np.array([float(v) for v in r.stdout.split()])
    assert len(predictions) == len(matrix), (
        f"saíram {len(predictions)} previsoes para {len(matrix)} linhas")
    return predictions


def compare(from_c, pred_sklearn, y, tolerance: float = TOLERANCE,
            max_swaps: float = MAX_SWAPS) -> dict:
    """Diferenca do C contra o sklearn. Exige ruido tipico e trocas de folha raras.

    Se qualquer um dos dois mudar de ordem de grandeza, e' bug de verdade (ordem das
    features trocada, constante errada, arvore mal gerada) — nao arredondamento.
    """
    diff = np.abs(np.asarray(from_c) - np.asarray(pred_sklearn))
    swaps = int((diff > tolerance).sum())
    assert diff.mean() < 1e-5, (
        f"diferenca media {diff.mean():.2e} kW - grande demais para ser float32")
    assert swaps <= max_swaps * len(from_c), (
        f"{swaps} de {len(from_c)} linhas trocaram de folha - esperado um punhado, nao isso")
    return {"linhas": len(from_c), "mediana": float(np.median(diff)),
            "media": float(diff.mean()), "maior": float(diff.max()), "trocas": swaps}


def measure_xtensa(obj_file: Path = paths.SCRATCH / "knob_senses_rf.o", *,
                   lib_src: Path = paths.LIB_RF_SRC,
                   impl: str = "knob_senses_rf.cpp") -> dict:
    """Compila o .cpp com o toolchain do ESP32-S3 e mede flash/RAM.

    O numero confiavel e' o do `size` SEM -A: somar as secoes a mao erra, porque o
    .literal (deposito de constantes float do Xtensa) e' ALLOC/LOAD e vai para o flash,
    mas nao se chama .text — somar so' text+rodata+data esquece ~25 KB.
    """
    obj_file = Path(obj_file)
    obj_file.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(f'"{XTENSA_GXX}" -O2 -std=c++17 -c '
                       f'"{Path(lib_src) / impl}" -o "{obj_file}" '
                       f'-I"{Path(lib_src)}"', shell=True, capture_output=True, text=True)
    assert r.returncode == 0, f"xtensa-g++ falhou:\n{r.stderr}"

    sections = {}
    r = subprocess.run(f'"{XTENSA_SIZE}" -A "{obj_file}"', shell=True,
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("."):
            sections[parts[0]] = int(parts[1])

    r = subprocess.run(f'"{XTENSA_SIZE}" "{obj_file}"', shell=True,
                       capture_output=True, text=True)
    fields = [ln for ln in r.stdout.splitlines() if ln.strip()][1].split()
    text, data, bss = int(fields[0]), int(fields[1]), int(fields[2])
    return {"flash": text + data, "ram": data + bss, "secoes": sections}
