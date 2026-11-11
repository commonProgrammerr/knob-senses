"""Envia o modelo MLP salvo ao Edge Impulse e baixa a lib C++ compilada.

Uso:
    python utils/exportar_mlp_ei.py 1h               # upload + build + download
    python utils/exportar_mlp_ei.py 1h --pular-deploy # download (deploy ja feito)
    python utils/exportar_mlp_ei.py 1h --seca        # valida os artefatos, sem rede

Pre-requisitos:
    - a celula de export do treino_mlp.ipynb rodou (dist/{freq}/modelo.tflite
      e dados_representativos.npy, gravados pelo mlp.export);
    - as chaves do Studio no .env da raiz do projeto (API_KEY e PROJET_ID;
      o .env NAO e' versionado) ou nas variaveis de ambiente EI_API_KEY e
      EI_PROJECT_ID, que tem prioridade sobre o .env;
    - pacote requests (ja' esta no venv do projeto).

Fluxo (API do Edge Impulse):
    0. quantizacao local: o SavedModel vira int8 (modelo_quantizado.tflite)
       com o TFLiteConverter e os dados representativos; o job do Studio nao
       quantiza o float enviado;
    1. POST /v1/api/{projeto}/jobs/deploy-pretrained-model
       (JSON com o .tflite quantizado em base64, o alvo "zip" (biblioteca
       C++) e o engine tflite em int8; header x-api-key) - nao exige
       impulso configurado
    2. GET  /v1/api/{projeto}/deployment/download?type=zip&modelType=int8
    O zip baixado substitui os diretorios gerados em lib/knob-senses/src/
    (edge-impulse-sdk, model-parameters e tflite-model), mantendo o
    library.json, o README e o CMakeLists do projeto.
"""
import argparse
import base64
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "models" / "src"))

from common import contract, paths
from mlp.transformacao import (
    gerar_header,
    verificar_modelo_embutido,
    verificar_sincronia,
)

API = "https://studio.edgeimpulse.com/v1/api"
PROJETO_PADRAO = "1160339"          # projeto do knob-senses no Edge Impulse
GERADOS = ("edge-impulse-sdk", "model-parameters", "tflite-model")


def _carregar_env() -> dict:
    """Le o .env da raiz (API_KEY, PROJET_ID), sem sobrescrever o ambiente.
    Tolerante a comentarios inline (texto apos ' #')."""
    caminho = RAIZ / ".env"
    if not caminho.exists():
        return {}
    valores = {}
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue
        chave, valor = linha.split("=", 1)
        valor = valor.strip().split(" #", 1)[0].strip().strip('"').strip("'")
        valores[chave.strip()] = valor
    return valores


def _sessao(api_key: str) -> requests.Session:
    s = requests.Session()
    s.headers["x-api-key"] = api_key
    return s


def _checa(r: requests.Response):
    """raise_for_status com o corpo da API no erro (ex.: papel da chave)."""
    try:
        r.raise_for_status()
    except requests.HTTPError:
        print(f"API respondeu HTTP {r.status_code}: {r.text[:400]}")
        raise


def _espera_job(sessao, projeto, job_id, rotulo, timeout_s=900):
    """Aguarda um job do Studio terminar (GET /jobs/{id}/status)."""
    url = f"{API}/{projeto}/jobs/{job_id}/status"
    t0 = time.monotonic()
    while True:
        r = sessao.get(url, timeout=60)
        _checa(r)
        dados = r.json()
        job = dados.get("job", dados)
        if job.get("finished"):
            if not job.get("finishedSuccessful"):
                raise RuntimeError(f"{rotulo}: job {job_id} terminou com erro: {dados}")
            print(f"{rotulo}: job {job_id} concluido")
            return
        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError(f"{rotulo}: job {job_id} nao terminou em {timeout_s}s")
        time.sleep(10)


def _quantizar(freq):
    """Converte o SavedModel para int8 com o TFLiteConverter local, usando os
    dados representativos do export. Devolve o caminho do .tflite quantizado.

    O deploy-pretrained-model do Studio NAO quantiza o float enviado: ele
    procura um model_quantized.tflite que so' existe no fluxo com impulso.
    Quantizando aqui, o int8 funciona sem impulso configurado.
    """
    import numpy as np
    import tensorflow as tf

    destino = paths.EXPORT_MLP / freq
    npy = destino / "dados_representativos.npy"
    if not npy.exists():
        raise FileNotFoundError(f"{npy} nao existe (rode a celula de export)")
    amostras = np.load(npy)

    def representativo():
        for i in range(len(amostras)):
            yield [amostras[i:i + 1].astype(np.float32)]

    conversor = tf.lite.TFLiteConverter.from_saved_model(
        str(destino / "saved_model"))
    conversor.optimizations = [tf.lite.Optimize.DEFAULT]
    conversor.representative_dataset = representativo
    conversor.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # E/S int8 (nao float32): o Studio so' reconhece o modelo como quantizado
    # quando os tensores de entrada e saida sao inteiros; com E/S float o job
    # procura um model_quantized.tflite que nunca e' gerado (ENOENT).
    conversor.inference_input_type = tf.int8
    conversor.inference_output_type = tf.int8

    caminho = destino / "modelo_quantizado.tflite"
    caminho.write_bytes(conversor.convert())
    return caminho


def _deploy(sessao, projeto, freq, seca, pular):
    """POST jobs/deploy-pretrained-model (tflite em base64, alvo zip
    (biblioteca C++), engine tflite, quantizacao int8) e aguarda. Nao exige
    impulso configurado no projeto."""
    if pular:
        print("deploy pulado (--pular-deploy): usando o ultimo build do projeto")
        return None
    destino = paths.EXPORT_MLP / freq
    float_ = destino / "modelo.tflite"
    if not float_.exists():
        raise FileNotFoundError(
            f"{float_} nao existe; rode a celula de export do treino_mlp.ipynb")
    if seca:
        print("seca: quantizacao e deploy nao executados")
        return None
    # Envia o modelo JA' QUANTIZADO: o job do Studio nao quantiza o float.
    tflite = _quantizar(freq)
    print(f"quantizado localmente: {tflite.name} "
          f"({tflite.stat().st_size / 1024:.1f} KB; float "
          f"{float_.stat().st_size / 1024:.1f} KB)")
    corpo = {
        "modelFileBase64": base64.b64encode(tflite.read_bytes()).decode("ascii"),
        "modelFileType": "tflite",
        "deploymentType": "zip",
        "engine": "tflite",
        "deployModelType": "int8",
        "modelInfo": {
            "input": {"inputType": "other"},
            "model": {"modelType": "regression"},
        },
    }
    npy = destino / "dados_representativos.npy"
    extra = ""
    if npy.exists():
        corpo["representativeFeaturesBase64"] = base64.b64encode(
            npy.read_bytes()).decode("ascii")
        extra = f" + {npy.name}"
    print(f"deploy: {tflite.relative_to(RAIZ)} (zip, tflite, int8){extra}")
    r = sessao.post(f"{API}/{projeto}/jobs/deploy-pretrained-model",
                    json=corpo, timeout=300)
    _checa(r)
    resp = r.json()
    if not resp.get("success"):
        raise RuntimeError(f"deploy falhou: {resp}")
    _espera_job(sessao, projeto, resp.get("id"), "deploy")
    return resp.get("id")


def _baixar(sessao, projeto, freq, seca):
    """GET deployment/download?type=zip; devolve o caminho do zip."""
    params = {"type": "zip", "modelType": "int8", "engine": "tflite"}
    print("download: deployment type=zip, modelType=int8, engine=tflite")
    if seca:
        return None
    r = sessao.get(f"{API}/{projeto}/deployment/download", params=params,
                   timeout=600)
    _checa(r)
    nome = "lib_edge_impulse.zip"
    cd = r.headers.get("Content-Disposition", "")
    if "filename=" in cd:
        nome = cd.split("filename=")[-1].strip().strip('"').strip("'")
    caminho = paths.EXPORT_MLP / freq / nome
    caminho.write_bytes(r.content)
    print(f"zip salvo: {caminho.relative_to(RAIZ)} ({len(r.content) / 1024:.0f} KB)")
    return caminho


def _instalar(caminho_zip):
    """Substitui os diretorios gerados em lib/knob-senses/src/, preservando o
    library.json, o README e o CMakeLists do projeto."""
    with zipfile.ZipFile(caminho_zip) as z:
        z.extractall(paths.EXPORT_MLP / "_ei_tmp")
    raiz_extraida = paths.EXPORT_MLP / "_ei_tmp"
    origem = {p.name: p for p in raiz_extraida.rglob("*")
              if p.is_dir() and p.name in GERADOS}
    faltam = [n for n in GERADOS if n not in origem]
    if faltam:
        raise RuntimeError(f"zip do Edge Impulse sem os diretorios esperados: {faltam}")
    for nome, pasta in origem.items():
        alvo = paths.LIB_MLP_SRC / nome
        if alvo.exists():
            shutil.rmtree(alvo)
        shutil.copytree(pasta, alvo)
    shutil.rmtree(raiz_extraida)
    print(f"lib atualizada: {paths.LIB_MLP.relative_to(RAIZ)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("freq", choices=contract.GRANULARIDADES,
                        help="granularidade do modelo (artefatos em dist/{freq}/)")
    parser.add_argument("--seca", action="store_true",
                        help="valida os artefatos locais e mostra o plano, sem rede")
    parser.add_argument("--pular-deploy", action="store_true",
                        help="nao dispara o deploy (usa o ultimo build zip)")
    args = parser.parse_args(argv)

    tflite = paths.EXPORT_MLP / args.freq / "modelo.tflite"
    if not tflite.exists():
        print(f"artefato ausente: {tflite}")
        print("Rode a celula de export do treino_mlp.ipynb antes.")
        return 1

    if args.seca:
        faltando = [
            p for p in (paths.EXPORT_MLP / args.freq / "dados_representativos.npy",)
            if not p.exists()
        ]
        if faltando:
            print(f"artefato ausente: {faltando[0]}")
            return 1
        ok, mensagem = verificar_sincronia(args.freq)
        print(mensagem)
        if not ok:
            return 1

    dotenv = _carregar_env()
    api_key = os.environ.get("EI_API_KEY") or dotenv.get("API_KEY")
    projeto = (os.environ.get("EI_PROJECT_ID") or dotenv.get("PROJET_ID")
               or PROJETO_PADRAO)
    if not args.seca and not api_key:
        print("defina API_KEY no .env da raiz (ou EI_API_KEY no ambiente)")
        return 1

    sessao = _sessao(api_key) if not args.seca else None
    _deploy(sessao, projeto, args.freq, args.seca, args.pular_deploy)
    caminho_zip = _baixar(sessao, projeto, args.freq, args.seca)
    if caminho_zip:
        _instalar(caminho_zip)
        gerar_header(args.freq)
        print(verificar_modelo_embutido(args.freq)
              or "aviso: modelo_quantizado.tflite ausente; verificacao SHA-256 pulada")
    if args.seca:
        print("seca: nada foi enviado nem baixado")
    else:
        print("OK: lib do Edge Impulse baixada e instalada em lib/knob-senses/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
