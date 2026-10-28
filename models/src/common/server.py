"""O protocolo do servidor de teste (v2): sessao unica, com numero de sequencia.

O firmware pede a amostra `seq`, infere e devolve a predicao da MESMA seq, que o
servidor grava em model_eval.csv - e' o banco de comparacao estimado x medido.

A sessao inteira vive em UMA conexao TCP, e o numero de sequencia e' a chave de tudo:
diz a que linha do df a mensagem pertence e volta ecoado em toda resposta. Perda nao
desalinha: conexao que morre volta ao accept() e o cliente reconecta reenviando a mesma
seq (predicao repetida e' ignorada, sem relogar). O fluxo completo esta' no serve().

AXES e' um alias de contract.RF_FEATURES de proposito: a ordem que o servidor transmite
e a ordem que o modelo espera, e o check_axis_order() confere isso contra o
normalizacao.json da lib entregue antes de transmitir qualquer coisa — ordem trocada NAO
da' erro, da' previsao errada em silencio.
"""

import json
import socket
import struct
from pathlib import Path

import numpy as np
import pandas as pd

from . import contract, paths

AXES = contract.RF_FEATURES         # alias: nao e' uma segunda lista

HOST = "0.0.0.0"
PORT = 3333

# Os bytes do protocolo v2 (ver README secao 4). O MESMO valor nos dois sentidos e'
# contrato, nao descuido: 0x01 e' REQUEST_SAMPLE no sentido ESP->servidor e SAMPLE no
# sentido contrario; 0x03 e' PREDICTION la' e ACK ca'. So' a direcao diz qual e'.
REQUEST_SAMPLE = 0x01   # ESP -> servidor: pede a amostra de `seq` (4 bytes u32 LE)
SAMPLE = 0x01           # servidor -> ESP: 0x01 + seq u32 + 4 float32 (um por AXES)
END_OF_DATA = 0x02      # servidor -> ESP: nao ha' seq >= len(df); o serve() retorna
PREDICTION = 0x03       # ESP -> servidor: a predicao da amostra `seq` (float32 LE)
ACK = 0x03              # servidor -> ESP: a linha `seq` esta' no log (duplicata: idem)

# Silencio maior que isto e' conexao morta/meia-aberta (ESP reiniciado sem FIN). 5 s >
# o timeout de leitura do firmware (2 s), entao quem desiste primeiro e' sempre o
# cliente: o servidor so' recolhe socket que o cliente ja' abandonou. Custo: o accept()
# fica preso ate' 5 s numa conexao zumbi (medido 4,8 s) - nesse meio-tempo o cliente
# retenta, e reenviar o pedido e' inofensivo (leitura pura, re-serve).
CONN_TIMEOUT = 5.0


def check_axis_order(path: Path = paths.LIB_RF / "normalizacao.json") -> list[str]:
    """Confere a ordem que o servidor transmite contra a que o modelo espera.

    Ordem errada nao da' erro: da' previsao errada em silencio. Por isso este assert
    roda ANTES de transmitir qualquer coisa.
    """
    order = json.loads(Path(path).read_text(encoding="utf-8"))
    assert tuple(order["features"]) == AXES, (
        "a ordem dos eixos nao bate com a do modelo: "
        f"{order['features']} (modelo) vs {AXES} (servidor). Reordenar aqui quebra a "
        "inferencia em silencio.")
    return list(order["features"])


def sample_bytes(sample) -> bytes:
    """A amostra como o firmware recebe: float32 LE, na ordem de AXES, sem cabecalho."""
    return np.array([sample[f] for f in AXES], dtype=np.float32).tobytes()


def open_socket(host: str = HOST, port: int = PORT) -> socket.socket:
    """O socket do servidor, com a protecao de porta do Windows.

    SO_EXCLUSIVEADDRUSE no Windows, e NAO SO_REUSEADDR. Medido nesta maquina:
      - com SO_REUSEADDR, um segundo processo consegue fazer bind na porta de um socket
        que ja esta escutando, em silencio: viram dois servidores na mesma porta e cada
        cliente cai em um.
      - SO_EXCLUSIVEADDRUSE bloqueia isso (WSAEADDRINUSE) e, medido tambem, ainda permite
        reabrir logo apos um cliente desconectar (TIME_WAIT).
    Nos outros sistemas SO_REUSEADDR e' o certo: la ele NAO permite sequestro, e e' ele
    que evita o TIME_WAIT.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
    if _exclusive is not None:
        srv.setsockopt(socket.SOL_SOCKET, _exclusive, 1)
    else:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((host, port))          # erro de porta estoura AQUI
        # listen(4), nao 1: medido nesta maquina, com a fila cheia (dreno de uma conexao
        # zumbi) o Windows RECUSA o SYN do cliente - fila rasa transforma a recuperacao
        # de queda numa rajada de connect recusado.
        srv.listen(4)
    except OSError:
        srv.close()
        raise
    # O timeout do accept() (0,5 s, so' para o Ctrl+C/Jupyter voltar ao bytecode mesmo
    # com o servidor ocioso) e' responsabilidade do serve(), que e' quem itera.
    return srv


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    """Le exatamente n bytes ou levanta.

    recv() num fluxo devolve o que tiver: os 4 bytes de um float podem chegar em dois
    pedacos. Ler "o que veio" era o que desalinhava a predicao do servidor antigo
    (recv(4) curto = float errado em silencio). EOF (b"") levanta; timeout do
    conn.settimeout() tambem - e quem chama trata os dois igual: a conexao morreu, o
    cliente reconecta e reenvia a seq.
    """
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionResetError(f"EOF com {len(buf)}/{n} bytes")
        buf += chunk
    return buf


def serve(df: pd.DataFrame, log_path: Path = paths.EVAL_LOG,
          srv: socket.socket | None = None) -> pd.DataFrame:
    """O laco do servidor: serve a amostra `seq` e loga a predicao confirmada.

    UMA conexao para a sessao inteira (o cliente mantem o socket). Mensagens de tamanho
    FIXO, little-endian (a tabela dos bytes esta' no README secao 4):

      ESP -> servidor  0x01 + seq u32             pede a amostra seq
      servidor -> ESP  0x01 + seq u32 + 4*f32     a amostra (grandeza BRUTA)
      ESP -> servidor  0x03 + seq u32 + f32       a predicao da amostra seq
      servidor -> ESP  0x03 + seq u32             ACK: a linha seq esta' no log
      servidor -> ESP  0x02 + seq u32             END_OF_DATA: seq >= len(df); retorna

    Duplicata e' idempotente: o ACK pode ter se perdido, entao o cliente reenvia a MESMA
    predicao - ela nao entra no log duas vezes e o ACK se repete. Conexao que morre no
    meio nao perde estado: o laco externo volta ao accept() e o cliente reconecta
    reenviando a seq em que parou.

    Termina quando o cliente pede uma amostra ALEM do fim (o END_OF_DATA fecha a sessao) e
    devolve o `reply` do painel. Se a maquina do cliente morrer antes, fica esperando -
    o CSV ja' gravado e' o fallback que o notebook le quando `reply is None`.

    LIMITACAO CONHECIDA: reiniciar serve() no meio do ensaio comeca um log VAZIO (o modo
    "w" trunca) e o cliente CONTINUA da seq dele - nao ha' estado em disco. O ensaio
    retoma do seq atual, mas as linhas anteriores se perdem. Para recomecar do zero:
    suba o servidor e de' reset no ESP (a seq volta a 0; o que ja' esta' logado volta
    como duplicata e nao e' relogado).
    """
    try:
        srv = srv if srv is not None else open_socket()
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        records: list[dict] = []        # uma entrada por predicao ACEITA, na ordem de chegada
        logged: set[int] = set()        # seq ja' no log: duplicata nao entra de novo
        last_duplicate: int | None = None
        n = len(df)

        # 0,5 s so' para o accept() VOLTAR ao bytecode: com o accept() bloqueado, o Ctrl+C
        # (parar a celula do Jupyter) fica pendente e so seria entregue quando chegasse um
        # cliente - ou seja, nunca, que e' justamente o estado ocioso do servidor.
        srv.settimeout(0.5)
        print(f"Server started at {srv.getsockname()[0]}:{srv.getsockname()[1]}", flush=True)

        with open(log_path, "w") as eval_log:
            eval_log.write("sample,real,pred,err\n")
            while True:                     # laco externo: tolera reconexao; o estado e' a seq
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue                # volta ao bytecode (o Ctrl+C e' entregue aqui)
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.settimeout(CONN_TIMEOUT)
                done = False
                with conn:
                    try:
                        while not done:
                            msg_type, seq = struct.unpack("<BI", _recv_exact(conn, 5))
                            if msg_type == REQUEST_SAMPLE:
                                if seq >= n:
                                    conn.sendall(struct.pack("<BI", END_OF_DATA, seq))
                                    print(f"fim do dataset: pediram a seq {seq}, ha' {n}",
                                        flush=True)
                                    done = True
                                else:
                                    # posicional (df.iloc), como o `for sample in df.iloc` de antes
                                    conn.sendall(struct.pack("<BI", SAMPLE, seq)
                                                + sample_bytes(df.iloc[seq]))
                            elif msg_type == PREDICTION:
                                (pred,) = struct.unpack("<f", _recv_exact(conn, 4))
                                if seq >= n:
                                    # Nao ha' linha para esta seq: dessincronia. Sem ACK, o
                                    # cliente trata como falha e retenta pelo caminho certo.
                                    print(f"predicao para seq fora do dataset: {seq}", flush=True)
                                    break
                                if seq not in logged:
                                    logged.add(seq)
                                    row = df.iloc[seq]
                                    real = row[contract.TARGET]
                                    # PCA == 0 e' sentinela de faltante, nao medicao: a linha
                                    # fica no log, mas fora do erro.
                                    real_valid = real != 0.0 and not np.isnan(real)
                                    record = dict(
                                        sample=row[contract.INDEX],
                                        pred=pred,
                                        real=real if real_valid else 0.0,
                                        err=abs(pred - real) if real_valid else 0.0,
                                    )
                                    records.append(record)
                                    # Ordem do CABECALHO (o reply do painel usa a ordem dele).
                                    eval_log.write(
                                        f"{record['sample']},{record['real']},"
                                        f"{record['pred']},{record['err']}\n")
                                    eval_log.flush()    # o CSV e' o fallback: que exista ja'
                                    # ACK ANTES do print: com a linha no log, o print (lento no
                                    # Jupyter) nao segura o cliente.
                                    conn.sendall(struct.pack("<BI", ACK, seq))
                                    print("sample %(sample)s: real=%(real).4f, pred=%(pred).4f, "
                                        "err=%(err).4f" % record, flush=True)
                                else:
                                    conn.sendall(struct.pack("<BI", ACK, seq))
                                    if seq != last_duplicate:
                                        # rajada (ex.: reboot do ESP) imprime uma vez so'
                                        print(f"seq {seq}: predicao duplicada (ACK se repete, "
                                            "sem relogar)", flush=True)
                                    last_duplicate = seq
                            else:
                                # 0x00 (o antigo PULA) e qualquer outro byte: fluxo fora do
                                # protocolo; fechar a conexao e' a unica saida segura.
                                print(f"mensagem desconhecida: tipo 0x{msg_type:02x}", flush=True)
                                break
                    except OSError:
                        pass                # cliente caiu/meia-aberta: volta ao accept
                if done:
                    break
        return pd.DataFrame(records, columns=["sample", "pred", "real", "err"])
    finally:
        srv.close()

def eval_metrics(df_eval: pd.DataFrame) -> dict:
    """R2/MAE/vies/amplitude do log do servidor — os numeros do painel de avaliacao."""
    np_real = df_eval["real"].to_numpy()
    np_pred = df_eval["pred"].to_numpy()
    ss_res = np.sum((np_pred - np_real) ** 2)
    ss_tot = ((np_real - np.mean(np_real)) ** 2).sum()
    teto = df_eval["pred"].max()
    return {
        "r2": 1 - (ss_res / ss_tot),
        "mae": float(df_eval["err"].mean()),
        "vies": float((df_eval["pred"] - df_eval["real"]).mean()),
        "teto": float(teto),
        "no_teto": int((df_eval["pred"] >= teto - 0.01).sum()),
        "amostras": len(df_eval),
    }
