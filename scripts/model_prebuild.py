"""Logica do pre-build dos modelos: regenera a lib do modelo (RF ou MLP)
somente quando o artefato treinado mudou.

Este modulo e' Python puro (sem SCons) para poder ser testado fora do
PlatformIO; o gancho do build esta em scripts/pre_build.py.

Artefato de disparo -> script que gera a lib:
    rf : dist/random_forest_{freq}.joblib  -> utils/gerar_lib_rf.py
    mlp: dist/{freq}/modelo.tflite         -> utils/exportar_mlp_ei.py

O carimbo fica em .pio/prebuild/{env}.json e guarda mtime_ns + tamanho do
artefato na ultima execucao bem-sucedida: mesma assinatura = nada a fazer.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

MODELOS = {
    "rf": {
        "modelo": "dist/random_forest_{freq}.joblib",
        "script": "utils/gerar_lib_rf.py",
        "alvo": "lib/knob-senses-rf/src/modelo_pca.h",
    },
    "mlp": {
        "modelo": "dist/{freq}/modelo.tflite",
        "script": "utils/exportar_mlp_ei.py",
        "alvo": "lib/knob-senses/src/tflite-model",
    },
}


def python_do_projeto(raiz: Path, preferido: str | None = None) -> str:
    """Python do venv do projeto (tem joblib/TensorFlow), com override."""
    if preferido:
        return preferido
    for rel in ("Scripts/python.exe", "bin/python"):
        candidato = raiz / ".venv" / rel
        if candidato.exists():
            return str(candidato)
    return sys.executable


def caminho_modelo(raiz: Path, env: str, freq: str) -> Path:
    return raiz / MODELOS[env]["modelo"].format(freq=freq)


def caminho_alvo(raiz: Path, env: str) -> Path:
    return raiz / MODELOS[env]["alvo"]


def carimbo(raiz: Path, env: str) -> Path:
    return raiz / ".pio" / "prebuild" / f"{env}.json"


def assinatura(modelo: Path) -> dict:
    """O que identifica a versao do artefato: mtime + tamanho."""
    st = modelo.stat()
    return {"mtime_ns": st.st_mtime_ns, "size": st.st_size}


def precisa_rodar(raiz: Path, env: str, freq: str) -> tuple[bool, str]:
    """(rodar?, motivo). Roda se falta artefato, falta lib ou o modelo mudou."""
    assert env in MODELOS, f"env sem pre-build: {env}"
    modelo = caminho_modelo(raiz, env, freq)
    alvo = caminho_alvo(raiz, env)
    if not modelo.exists():
        return True, f"artefato ausente ({modelo.relative_to(raiz)})"
    if not alvo.exists():
        return True, f"lib ausente ({alvo.relative_to(raiz)})"
    arq = carimbo(raiz, env)
    if not arq.exists():
        return True, "primeira execucao (sem carimbo)"
    try:
        anterior = json.loads(arq.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True, "carimbo ilegivel"
    if anterior.get("freq") != freq:
        return True, f"granularidade mudou ({anterior.get('freq')} -> {freq})"
    if anterior.get("assinatura") != assinatura(modelo):
        return True, "modelo mudou desde a ultima lib"
    return False, "modelo inalterado"


def executar(raiz: Path, env: str, freq: str, python: str | None = None) -> None:
    """Roda o script do modelo e grava o carimbo (so' se ele passar)."""
    script = raiz / MODELOS[env]["script"]
    cmd = [python_do_projeto(raiz, python), str(script), freq]
    print(f"[prebuild] executando: {' '.join(cmd)}", flush=True)
    resultado = subprocess.run(cmd, cwd=raiz)
    if resultado.returncode != 0:
        raise RuntimeError(
            f"[prebuild] {script.name} falhou (codigo {resultado.returncode}); "
            "o build foi interrompido para nao embarcar uma lib velha")
    arq = carimbo(raiz, env)
    arq.parent.mkdir(parents=True, exist_ok=True)
    modelo = caminho_modelo(raiz, env, freq)
    arq.write_text(json.dumps({
        "modelo": str(modelo.relative_to(raiz)),
        "freq": freq,
        "assinatura": assinatura(modelo),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[prebuild] lib do {env} regenerada; carimbo em {arq.relative_to(raiz)}",
          flush=True)
