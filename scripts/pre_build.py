"""Gancho do PlatformIO: pre-build dos modelos RF e MLP.

Ligado em platformio.ini nos dois envs:

    extra_scripts = pre:scripts/pre_build.py

Antes de compilar, roda utils/gerar_lib_rf.py (rf) ou utils/exportar_mlp_ei.py
(mlp) SOMENTE se o artefato treinado mudou desde a ultima lib gerada. A
decisao e' do scripts/model_prebuild.py (assinatura mtime+tamanho).

Opcoes de projeto (platformio.ini):
    custom_rf_freq  = 1h   granularidade embarcada do RF
    custom_mlp_freq = 1h   granularidade embarcada do MLP
    custom_mlp_quantization = int8  quantization do MLP ("int8" | "float32" | ...)
    custom_python   = ...  python alternativo (default: .venv do projeto)

Para pular o pre-build (ex.: sem rede para o Edge Impulse):
    set KNOB_SKIP_PREBUILD=1   (Windows)   /   export KNOB_SKIP_PREBUILD=1
"""
import os
import sys
from pathlib import Path

Import("env")

RAIZ = Path(env.subst("$PROJECT_DIR"))
ENV = env.subst("$PIOENV")

sys.path.insert(0, str(RAIZ / "scripts"))
import model_prebuild as mp  # noqa: E402


def _freq() -> str:
    opcao = "custom_rf_freq" if ENV == "rf" else "custom_mlp_freq"
    return env.GetProjectOption(opcao, "1h")


def _quantization() -> str:
    return env.GetProjectOption("custom_mlp_quantization", "int8")


def _pre_build():
    if ENV not in mp.MODELOS:
        return
    freq = _freq()
    quantization = _quantization()
    if os.environ.get("KNOB_SKIP_PREBUILD"):
        header = mp.caminho_header(RAIZ, ENV)
        if header is not None and not header.exists():
            print(f"[prebuild] AVISO: {header.relative_to(RAIZ)} ausente; "
                  f"rode utils/gerar_header_mlp.py {freq}")
        print(f"[prebuild] {ENV}: pulado por KNOB_SKIP_PREBUILD")
        return
    acao, motivo = mp.precisa_rodar(RAIZ, ENV, freq, quantization)
    if acao == "nada":
        print(f"[prebuild] {ENV} ({freq}, {quantization}): {motivo}; artefatos mantidos")
        return
    print(f"[prebuild] {ENV} ({freq}, {quantization}): {motivo}")
    mp.executar(RAIZ, ENV, freq, acao=acao,
                python=env.GetProjectOption("custom_python", "") or None,
                quantization=quantization)


# Roda na CARGA do script, nao como pre-action do SCons: AddPreAction so'
# dispara quando o SCons decide recompilar algum alvo, e com o build em dia a
# checagem do modelo nunca rodaria. O PlatformIO carrega os extra_scripts
# (prefixo "pre:") em TODA invocacao, antes de montar o grafo de build - e'
# o ponto que garante a checagem antes de qualquer compilacao.
_pre_build()
