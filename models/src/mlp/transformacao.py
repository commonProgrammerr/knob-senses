"""Constantes da transformacao do alvo do MLP, geradas dos JSONs do treino.

O firmware do MLP precisa de tres coisas que o SDK do Edge Impulse nao entrega:
o min-max das entradas (o impulso nao normaliza), os parametros da inversa de
Box-Cox e o offset de calibracao. Este modulo le os dois JSONs que o treino
grava em dist/{freq}/, valida o contrato e escreve o header C++ que o firmware
inclui. E' Python puro: nao importa TensorFlow nem fala com o Edge Impulse.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from pathlib import Path

from common import contract, paths
from common.templates import fill_template

HEADER = paths.ROOT / "include" / "mlp_transformacao_alvo.h"
_ORDEM_ESPERADA = (
    "clip_boxcox -> inv_boxcox -> - shift_kw -> + offset_kw "
    "-> clip(pca_min_kw, pca_max_kw)"
)

_TEMPLATE = """/*
 * knob-senses - constantes da transformacao do MLP (potencia CA, kW).
 *
 * GERADO POR utils/gerar_header_mlp.py a partir de:
 *   dist/@@FREQ@@/normalizacao.json
 *   dist/@@FREQ@@/transformacao_alvo.json
 * NAO EDITE A MAO: a proxima geracao sobrescreve este arquivo.
 *
 * Ordem do pos-processamento:
 *   clip_boxcox -> inv_boxcox -> - shift_kw -> + offset_kw
 *   -> clip(pca_min_kw, pca_max_kw)
 */
#ifndef KNOB_MLP_TRANSFORMACAO_ALVO_H
#define KNOB_MLP_TRANSFORMACAO_ALVO_H

#include <stddef.h>

constexpr size_t KNOB_MLP_N_FEATURES = @@N_FEATURES@@;
#define KNOB_MLP_ORDEM "@@ORDEM@@"
#define KNOB_MLP_FREQ  "@@FREQ@@"

/* Min-max ajustado SO no treino (2021..2024H1), na ordem acima. */
constexpr float KNOB_MLP_MINIMO[KNOB_MLP_N_FEATURES] = {
  @@MINIMO@@
};
constexpr float KNOB_MLP_MAXIMO[KNOB_MLP_N_FEATURES] = {
  @@MAXIMO@@
};

/* Shift + Box-Cox, ajustados SO no treino. */
constexpr float KNOB_MLP_SHIFT_KW        = @@SHIFT@@;
constexpr float KNOB_MLP_BOXCOX_LAMBDA   = @@LAMBDA@@;
constexpr float KNOB_MLP_BOXCOX_CLIP_MIN = @@CLIP_MIN@@;
constexpr float KNOB_MLP_BOXCOX_CLIP_MAX = @@CLIP_MAX@@;

/* Calibracao de vies (2024H2) e faixa fisica do alvo. */
constexpr float KNOB_MLP_OFFSET_KW       = @@OFFSET@@;
constexpr float KNOB_MLP_PCA_MIN_KW      = @@PCA_MIN@@;
constexpr float KNOB_MLP_PCA_MAX_KW      = @@PCA_MAX@@;

static_assert(1.0f + KNOB_MLP_BOXCOX_LAMBDA * KNOB_MLP_BOXCOX_CLIP_MIN > 0.0f,
              "clip_boxcox inferior fora do dominio da inversa");
static_assert(1.0f + KNOB_MLP_BOXCOX_LAMBDA * KNOB_MLP_BOXCOX_CLIP_MAX > 0.0f,
              "clip_boxcox superior fora do dominio da inversa");

#endif /* KNOB_MLP_TRANSFORMACAO_ALVO_H */
"""


def _json(freq: str, nome: str) -> dict:
    caminho = paths.EXPORT_MLP / freq / nome
    try:
        return json.loads(caminho.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{caminho} nao existe; rode o treino do MLP para {freq}"
        ) from None
    except ValueError as e:
        raise ValueError(f"{caminho} nao e' JSON valido: {e}") from None


def _num(valor, nome: str) -> float:
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise ValueError(f"{nome}: esperado numero, veio {valor!r}")
    valor = float(valor)
    if not math.isfinite(valor):
        raise ValueError(f"{nome}: valor nao finito ({valor!r})")
    return valor


def _float32(valor: float) -> float:
    return struct.unpack("<f", struct.pack("<f", valor))[0]


def _fmt_float(valor: float) -> str:
    texto = f"{valor:.9g}"
    if "e" not in texto and "E" not in texto and "." not in texto:
        texto += ".0"
    return texto + "f"


def _validar_features(norm: dict, trans: dict) -> tuple[str, ...]:
    ordem = tuple(norm.get("ordem_das_entradas", ()))
    if ordem != contract.RF_FEATURES:
        raise ValueError(
            f"ordem_das_entradas={ordem} nao bate com "
            f"contract.RF_FEATURES={contract.RF_FEATURES}"
        )
    if tuple(norm.get("features", ())) != contract.RF_FEATURES:
        raise ValueError(
            f"normalizacao.json['features']={norm.get('features')!r} "
            "nao bate com o contrato"
        )
    if tuple(trans.get("features", ())) != contract.RF_FEATURES:
        raise ValueError(
            f"transformacao_alvo.json['features']={trans.get('features')!r} "
            "nao bate com o contrato"
        )
    return contract.RF_FEATURES


def carregar_e_validar(freq: str) -> dict:
    """Le e valida os dois JSONs; devolve as constantes na ordem do contrato."""
    if freq not in contract.GRANULARIDADES:
        raise ValueError(f"freq={freq!r} fora de {contract.GRANULARIDADES}")

    norm = _json(freq, "normalizacao.json")
    trans = _json(freq, "transformacao_alvo.json")
    features = _validar_features(norm, trans)

    if norm.get("normalizacao") != "min_max":
        raise ValueError("normalizacao.json: esperado normalizacao='min_max'")
    if trans.get("freq") != freq:
        raise ValueError(
            f"transformacao_alvo.json: freq={trans.get('freq')!r} != {freq!r}"
        )

    minimo = norm.get("minimo")
    maximo = norm.get("maximo")
    if not isinstance(minimo, dict) or not isinstance(maximo, dict):
        raise ValueError("normalizacao.json: minimo/maximo devem ser dicionarios")
    if set(minimo) != set(features) or set(maximo) != set(features):
        raise ValueError(
            "normalizacao.json: minimo/maximo com chaves diferentes das features"
        )
    vmin = [_num(minimo[f], f"minimo[{f}]") for f in features]
    vmax = [_num(maximo[f], f"maximo[{f}]") for f in features]
    for i, f in enumerate(features):
        if vmax[i] <= vmin[i]:
            raise ValueError(f"normalizacao.json: maximo[{f}] <= minimo[{f}]")

    shift = _num(trans.get("shift_kw"), "shift_kw")
    lam = _num(trans.get("lambda"), "lambda")
    clip = trans.get("clip_boxcox")
    if not isinstance(clip, list) or len(clip) != 2:
        raise ValueError("transformacao_alvo.json: clip_boxcox deve ter 2 valores")
    clip_min = _num(clip[0], "clip_boxcox[0]")
    clip_max = _num(clip[1], "clip_boxcox[1]")
    if clip_min >= clip_max:
        raise ValueError("transformacao_alvo.json: clip_boxcox fora de ordem")

    offset = _num(trans.get("offset_kw"), "offset_kw")
    pca_min = _num(trans.get("pca_min_kw"), "pca_min_kw")
    pca_max = _num(trans.get("pca_max_kw"), "pca_max_kw")
    if not (0.0 <= pca_min < pca_max <= 2.0):
        raise ValueError("transformacao_alvo.json: faixa fisica invalida")

    if abs(lam) > 1e-12:
        if abs(lam) < 1e-6:
            raise ValueError("transformacao_alvo.json: lambda degenerado")
        for nome, z in (("clip_min", clip_min), ("clip_max", clip_max)):
            base = 1.0 + lam * z
            if base <= 0.0:
                raise ValueError(
                    f"transformacao_alvo.json: 1 + lambda*{nome} <= 0"
                )
            base32 = _float32(
                _float32(1.0) + _float32(_float32(lam) * _float32(z))
            )
            if base32 <= 0.0:
                raise ValueError(
                    f"transformacao_alvo.json: 1 + lambda*{nome} <= 0 em float32"
                )

    if "calibracao_offset_kw" not in norm:
        raise ValueError("normalizacao.json: falta calibracao_offset_kw")
    if not math.isclose(
        _num(norm["calibracao_offset_kw"], "calibracao_offset_kw"),
        offset,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError("normalizacao.json: calibracao_offset_kw != offset_kw")

    alvo = norm.get("alvo")
    if not isinstance(alvo, dict):
        raise ValueError("normalizacao.json: falta alvo")
    if not math.isclose(
        _num(alvo.get("min"), "alvo.min"), pca_min, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError("normalizacao.json: alvo.min != pca_min_kw")
    if not math.isclose(
        _num(alvo.get("max"), "alvo.max"), pca_max, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError("normalizacao.json: alvo.max != pca_max_kw")

    reduzida = norm.get("transformacao_alvo")
    if not isinstance(reduzida, dict):
        raise ValueError("normalizacao.json: falta transformacao_alvo")
    if "boxcox" not in str(reduzida.get("tipo", "")).lower():
        raise ValueError("normalizacao.json: transformacao_alvo.tipo nao e' Box-Cox")
    for campo, esperado in (("shift_kw", shift), ("lambda", lam)):
        if not math.isclose(
            _num(reduzida.get(campo), f"transformacao_alvo.{campo}"),
            esperado,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"normalizacao.json: transformacao_alvo.{campo} difere do sidecar"
            )
    reduzida_clip = reduzida.get("clip_boxcox")
    if not isinstance(reduzida_clip, list) or len(reduzida_clip) != 2:
        raise ValueError(
            "normalizacao.json: transformacao_alvo.clip_boxcox invalido"
        )
    for i, esperado in enumerate((clip_min, clip_max)):
        if not math.isclose(
            _num(reduzida_clip[i], f"transformacao_alvo.clip_boxcox[{i}]"),
            esperado,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "normalizacao.json: transformacao_alvo.clip_boxcox difere do sidecar"
            )

    ordem = " ".join(str(trans.get("pos_processamento", "")).split())
    if ordem != _ORDEM_ESPERADA:
        raise ValueError(
            f"transformacao_alvo.json: pos_processamento={ordem!r} "
            f"!= {_ORDEM_ESPERADA!r}"
        )

    return {
        "freq": freq,
        "features": features,
        "minimo": vmin,
        "maximo": vmax,
        "shift_kw": shift,
        "lambda": lam,
        "clip_min": clip_min,
        "clip_max": clip_max,
        "offset_kw": offset,
        "pca_min_kw": pca_min,
        "pca_max_kw": pca_max,
    }


def verificar_modelo_embutido(
    freq: str, modelo_nome: str = "modelo_quantizado.tflite"
) -> str | None:
    """Confere o modelo embutido na lib contra dist/{freq}/<modelo_nome>.

    O padrao e' o int8; o deploy float32 passa "modelo.tflite". Aceita os dois
    formatos do Edge Impulse: array xxd no header ou .tflite separado ligado por
    INCBIN. Devolve None quando o .tflite local nao existe.
    """
    local = paths.EXPORT_MLP / freq / modelo_nome
    if not local.exists():
        return None
    pasta = paths.LIB_MLP_SRC / "tflite-model"
    headers = sorted(pasta.glob("tflite_learn_*.h"))
    if len(headers) != 1:
        raise RuntimeError(
            f"esperado 1 header tflite_learn_*.h em {pasta}, "
            f"achei {len(headers)}")
    texto = headers[0].read_text(encoding="utf-8", errors="ignore")
    corpo = re.search(
        r"const unsigned char tflite_learn_[A-Za-z0-9_]+\[\]\s*=\s*\{(.*?)\};",
        texto, re.S)
    if corpo is not None:
        embutido = bytes(
            int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]{2})", corpo.group(1)))
        origem = headers[0].name
    else:
        binarios = sorted(pasta.glob("tflite_learn_*.tflite"))
        if len(binarios) != 1:
            raise RuntimeError(
                f"modelo nao encontrado em {headers[0]}: nem array xxd, "
                f"nem .tflite unico em {pasta}")
        embutido = binarios[0].read_bytes()
        origem = binarios[0].name
    esperado = local.read_bytes()
    if hashlib.sha256(embutido).digest() != hashlib.sha256(esperado).digest():
        raise RuntimeError(
            f"a lib instalada nao e' o modelo de {freq}: "
            f"{origem} tem {len(embutido)} bytes, "
            f"{local.name} tem {len(esperado)} bytes")
    return f"modelo confere com {local.relative_to(paths.ROOT)} ({origem})"


def render_header(freq: str, dados: dict) -> str:
    """Renderiza o header de forma deterministica (sem timestamp)."""
    return fill_template(
        _TEMPLATE,
        FREQ=freq,
        N_FEATURES=len(dados["features"]),
        ORDEM=", ".join(dados["features"]),
        MINIMO=",\n  ".join(_fmt_float(v) for v in dados["minimo"]),
        MAXIMO=",\n  ".join(_fmt_float(v) for v in dados["maximo"]),
        SHIFT=_fmt_float(dados["shift_kw"]),
        LAMBDA=_fmt_float(dados["lambda"]),
        CLIP_MIN=_fmt_float(dados["clip_min"]),
        CLIP_MAX=_fmt_float(dados["clip_max"]),
        OFFSET=_fmt_float(dados["offset_kw"]),
        PCA_MIN=_fmt_float(dados["pca_min_kw"]),
        PCA_MAX=_fmt_float(dados["pca_max_kw"]),
    )


def gerar_header(freq: str, destino: Path | None = None) -> Path:
    """Valida os JSONs e escreve o header (so' se o conteudo mudou)."""
    dados = carregar_e_validar(freq)
    texto = render_header(freq, dados)
    destino = Path(destino) if destino is not None else HEADER
    if destino.exists() and destino.read_text(encoding="utf-8") == texto:
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(destino.name + ".tmp")
    tmp.write_text(texto, encoding="utf-8", newline="\n")
    tmp.replace(destino)
    return destino


def verificar_sincronia(freq: str, destino: Path | None = None) -> tuple[bool, str]:
    """Compara o header existente com o renderizado; nao escreve."""
    dados = carregar_e_validar(freq)
    texto = render_header(freq, dados)
    destino = Path(destino) if destino is not None else HEADER
    if not destino.exists():
        return False, f"header ausente: {destino}"
    if destino.read_text(encoding="utf-8") != texto:
        return (
            False,
            f"header dessincronizado: {destino}; "
            f"rode utils/gerar_header_mlp.py {freq}",
        )
    return True, f"header em dia: {destino}"


__all__ = [
    "HEADER",
    "carregar_e_validar",
    "render_header",
    "gerar_header",
    "verificar_sincronia",
    "verificar_modelo_embutido",
]
