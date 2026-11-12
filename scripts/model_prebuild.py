"""Logica do pre-build dos modelos: regenera a lib do modelo (RF ou MLP)
somente quando o artefato treinado mudou.

Este modulo e' Python puro (sem SCons) para poder ser testado fora do
PlatformIO; o gancho do build esta em scripts/pre_build.py.

Artefato de disparo -> script que gera a lib:
    rf : dist/random_forest_{freq}.joblib  -> utils/gerar_lib_rf.py
    mlp: dist/{freq}/modelo.tflite + dados_representativos.npy
         -> utils/exportar_mlp_ei.py

O MLP tem dois artefatos de saida: a lib do Edge Impulse (depende do modelo e
dos dados representativos) e o header de transformacao (depende dos JSONs de
normalizacao). O carimbo guarda as duas assinaturas; se so' os JSONs mudarem,
o pre-build roda apenas utils/gerar_header_mlp.py, sem rede.

O carimbo fica em .pio/prebuild/{env}.json e guarda mtime_ns + tamanho dos
artefatos na ultima execucao bem-sucedida: mesma assinatura = nada a fazer.
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
    "mlp-ei": {
        "modelo": "dist/{freq}/modelo.tflite",
        "representativos": "dist/{freq}/dados_representativos.npy",
        "transformacao": (
            "dist/{freq}/normalizacao.json",
            "dist/{freq}/transformacao_alvo.json",
        ),
        "script": "utils/exportar_mlp_ei.py",
        "script_header": "utils/gerar_header_mlp.py",
        "alvo": "lib/knob-senses/src/tflite-model",
        "header": "include/mlp_transformacao_alvo.h",
    },
    "mlp-tflite": {
        "modelo": "dist/{freq}/modelo.tflite",
        "representativos": "dist/{freq}/dados_representativos.npy",
        "transformacao": (
            "dist/{freq}/normalizacao.json",
            "dist/{freq}/transformacao_alvo.json",
        ),
        "script": "utils/generate_model_data_mlp.py",
        "script_header": "utils/gerar_header_mlp.py",
        "alvo": "lib/knob-mlp-tflite/src/model_data.cc",
        "header": "include/mlp_transformacao_alvo.h",
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


def _caminho(raiz: Path, padrao: str, freq: str) -> Path:
    return raiz / padrao.format(freq=freq)


def caminho_modelo(raiz: Path, env: str, freq: str) -> Path:
    return _caminho(raiz, MODELOS[env]["modelo"], freq)


def caminho_alvo(raiz: Path, env: str) -> Path:
    return raiz / MODELOS[env]["alvo"]


def caminho_header(raiz: Path, env: str) -> Path | None:
    padrao = MODELOS[env].get("header")
    return raiz / padrao if padrao else None


def caminhos_modelo(raiz: Path, env: str, freq: str) -> list[Path]:
    """Artefatos que mudam o modelo embarcado (exigem export completo)."""
    cfg = MODELOS[env]
    caminhos = [caminho_modelo(raiz, env, freq)]
    if "representativos" in cfg:
        caminhos.append(_caminho(raiz, cfg["representativos"], freq))
    return caminhos


def caminhos_transformacao(raiz: Path, env: str, freq: str) -> list[Path]:
    """JSONs que so' mudam o header de pos-processamento."""
    return [
        _caminho(raiz, padrao, freq)
        for padrao in MODELOS[env].get("transformacao", ())
    ]


def carimbo(raiz: Path, env: str) -> Path:
    return raiz / ".pio" / "prebuild" / f"{env}.json"


def assinatura(caminho: Path) -> dict:
    """O que identifica a versao do artefato: mtime + tamanho."""
    st = caminho.stat()
    return {"mtime_ns": st.st_mtime_ns, "size": st.st_size}


def assinatura_conjunto(raiz: Path, caminhos: list[Path]) -> dict:
    return {
        str(caminho.relative_to(raiz)): assinatura(caminho)
        for caminho in caminhos
    }


def precisa_rodar(raiz: Path, env: str, freq: str,
                  quantization: str = "int8") -> tuple[str, str]:
    """Devolve (acao, motivo): 'completo', 'header' ou 'nada'."""
    assert env in MODELOS, f"env sem pre-build: {env}"
    modelo = caminho_modelo(raiz, env, freq)
    alvo = caminho_alvo(raiz, env)
    header = caminho_header(raiz, env)

    if not modelo.exists():
        return "completo", f"artefato ausente ({modelo.relative_to(raiz)})"
    if not alvo.exists():
        return "completo", f"lib ausente ({alvo.relative_to(raiz)})"
    for caminho in caminhos_modelo(raiz, env, freq):
        if not caminho.exists():
            return "completo", f"artefato ausente ({caminho.relative_to(raiz)})"

    arq = carimbo(raiz, env)
    if not arq.exists():
        return "completo", "primeira execucao (sem carimbo)"
    try:
        anterior = json.loads(arq.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "completo", "carimbo ilegivel"
    if anterior.get("freq") != freq:
        return "completo", f"granularidade mudou ({anterior.get('freq')} -> {freq})"
    if anterior.get("quantization", "int8") != quantization:
        return "completo", (
            f"quantization mudou ({anterior.get('quantization', 'int8')} -> {quantization})")

    atual_modelo = assinatura_conjunto(
        raiz, caminhos_modelo(raiz, env, freq))
    anterior_modelo = anterior.get("assinatura_modelo")
    if anterior_modelo is None:
        # Carimbo antigo: {"assinatura": {...}} do unico .tflite/.joblib.
        legado = anterior.get("assinatura")
        if legado is None or legado != assinatura(modelo):
            return "completo", "carimbo antigo sem assinatura compativel"
        if header is None:
            return "nada", "modelo inalterado (carimbo antigo)"
        return "header", "carimbo antigo; gerando so' o header"
    if anterior_modelo != atual_modelo:
        return "completo", "modelo mudou desde a ultima lib"

    if header is not None and not header.exists():
        return "header", f"header ausente ({header.relative_to(raiz)})"
    if header is None:
        return "nada", "modelo inalterado"

    atual_transformacao = assinatura_conjunto(
        raiz, caminhos_transformacao(raiz, env, freq))
    if anterior.get("assinatura_transformacao") != atual_transformacao:
        return "header", "transformacao mudou desde o ultimo header"
    return "nada", "modelo e header inalterados"


def executar(raiz: Path, env: str, freq: str, acao: str,
             python: str | None = None, quantization: str = "int8") -> None:
    """Roda o script da acao e grava o carimbo (so' se ele passar)."""
    assert acao in ("completo", "header"), f"acao invalida: {acao}"
    cfg = MODELOS[env]
    if acao == "completo":
        rel = cfg["script"]
    else:
        rel = cfg.get("script_header")
        if not rel:
            raise RuntimeError(f"{env} nao tem script_header para a acao 'header'")
    script = raiz / rel
    cmd = [python_do_projeto(raiz, python), str(script), freq]
    if env == "mlp-ei":
        if quantization == "float32":
            cmd.append("--float32")
    else:
        cmd.append(quantization)
    print(f"[prebuild] executando: {' '.join(cmd)}", flush=True)
    resultado = subprocess.run(cmd, cwd=raiz)
    if resultado.returncode != 0:
        raise RuntimeError(
            f"[prebuild] {script.name} falhou (codigo {resultado.returncode}); "
            "o build foi interrompido para nao embarcar um artefato velho")
    arq = carimbo(raiz, env)
    arq.parent.mkdir(parents=True, exist_ok=True)
    modelo = caminho_modelo(raiz, env, freq)
    dados = {
        "modelo": str(modelo.relative_to(raiz)),
        "freq": freq,
        "quantization": quantization,
        "assinatura_modelo": assinatura_conjunto(
            raiz, caminhos_modelo(raiz, env, freq)),
    }
    if cfg.get("header"):
        dados["assinatura_transformacao"] = assinatura_conjunto(
            raiz, caminhos_transformacao(raiz, env, freq))
    arq.write_text(json.dumps(dados, indent=2) + "\n", encoding="utf-8")
    print(f"[prebuild] {env} atualizado; carimbo em {arq.relative_to(raiz)}",
          flush=True)
