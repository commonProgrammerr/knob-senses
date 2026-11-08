"""O protocolo do servidor de teste (v2): sessao unica, com numero de sequencia.

O firmware pede a amostra `seq`, infere e devolve a predicao da MESMA seq, que o
servidor grava no log do ensaio - e' o banco de comparacao estimado x medido.

O transporte e' a CDC do USB-Serial/JTAG do ESP32-S3 (COM29); o log do firmware fica
na UART0 (COM28). A sessao comeca com um handshake de 1 byte (MAGIC/MAGIC_ACK, nas
variantes RESET e RESUME) que deixa para tras o lixo de boot do ROM. O fluxo e' em
LOTES: o host manda um lote de amostras (0x01 + base + count + payload) e recebe um
lote de predicoes (0x03 + base + count + payload) - um par de transferencias por lote
(medido nesta bancada: o downlink deste elo tem soluços de ~2 s a cada ~30 mensagens
pequenas, e o lote amortiza cada soluço por ~64 linhas). O lote e' idempotente:
leitura que estoura o timeout refaz o handshake e o host reenvia o MESMO lote. O
fluxo completo esta' no serve_serial().

AXES e' um alias de contract.RF_FEATURES de proposito: a ordem que o servidor transmite
e a ordem que o modelo espera, e o check_axis_order() confere isso contra o
normalizacao.json da lib entregue antes de transmitir qualquer coisa — ordem trocada NAO
da' erro, da' previsao errada em silencio.
"""

import json
import struct
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import contract, paths

AXES = contract.RF_FEATURES         # alias: nao e' uma segunda lista

# Os bytes do protocolo v2 (ver README secao 4). O MESMO valor nos dois sentidos e'
# contrato, nao descuido: 0x01 e' REQUEST_SAMPLE no sentido ESP->servidor e SAMPLE no
# sentido contrario; 0x03 e' PREDICTION la' e ACK ca'. So' a direcao diz qual e'.
REQUEST_SAMPLE = 0x01   # ESP -> servidor: pede a amostra de `seq` (4 bytes u32 LE)
SAMPLE = 0x01           # servidor -> ESP: 0x01 + seq u32 + 4 float32 (um por AXES)
END_OF_DATA = 0x02      # servidor -> ESP: nao ha' seq >= len(df); o serve() retorna
PREDICTION = 0x03       # ESP -> servidor: a predicao da amostra `seq` (float32 LE)
ACK = 0x03              # servidor -> ESP: a linha `seq` esta' no log (duplicata: idem)

# --------------------------------------------------------- transporte serial
# A CDC do USB-Serial/JTAG (COM29) carrega os dados do ensaio; o log do firmware
# fica na UART0 (COM28). Mesmo protocolo v2, outro meio. O MAGIC absorve o lixo
# de boot do ROM (o ESP escreve na CDC antes de nos) e marca o inicio da sessao;
# o MAGIC de volta fecha o handshake. Duas variantes: RESET (nova usina: a seq
# do firmware volta a 0) e RESUME (re-handshake da MESMA sessao: a seq fica onde
# esta - sem isso, um soluço faria o firmware re-percorrer o dataset do zero).
# Sem resposta, o MAGIC e' reenviado: o firmware pode ter consumido um como
# parte de uma troca interrompida.
MAGIC = 0x4B          # RESUME, e o byte de volta (MAGIC_ACK) do firmware
MAGIC_RESET = 0x52    # RESET: a seq do firmware volta a 0
MAGIC_RETRIES = 5
MAGIC_INTERVAL_S = 2.0
# 5 s > os 2 s do firmware: quem desiste primeiro e' SEMPRE o firmware (como no
# TCP antigo) - o host aguenta um soluço e reencontra o cliente pelo handshake.
SERIAL_READ_TIMEOUT_S = 5.0

# Amostras por lote do protocolo v3: 777 B no sentido host->ESP, 265 B na volta.
BATCH = 64


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


def _record_line(record: dict) -> str:
    """A linha de progresso do painel, identica nos dois transportes."""
    return "sample %(sample)s: real=%(real).4f, pred=%(pred).4f, err=%(err).4f" % record


def _log_prediction(df, seq, pred, logged, records, eval_log) -> dict | None:
    """Grava a predicao `seq` (uma vez: duplicata so' re-ACK) e devolve o registro.

    PCA == 0 e' sentinela de faltante e PCA NaN e' gap artificial: a linha fica no
    log com real/err NaN, fora do erro (quem mede o erro DENTRO dos gaps junta o
    log com o dataset original).
    """
    if seq in logged:
        return None
    logged.add(seq)
    row = df.iloc[seq]
    real = row[contract.TARGET]
    real_valid = real != 0.0 and not np.isnan(real)
    record = dict(
        sample=row[contract.INDEX],
        pred=pred,
        real=real if real_valid else float("nan"),
        err=abs(pred - real) if real_valid else float("nan"),
    )
    records.append(record)
    # Ordem do CABECALHO (o reply do painel usa a ordem dele).
    eval_log.write(
        f"{record['sample']},{record['real']},{record['pred']},{record['err']}\n")
    eval_log.flush()    # o CSV e' o fallback: que exista ja'
    return record


def _serial_read_exact(ser, n: int, timeout_s: float = SERIAL_READ_TIMEOUT_S) -> bytes:
    """Le exatamente n bytes da serial ou levanta.

    read() devolve o que tiver: os 4 bytes de um float podem chegar em pedacos,
    e "ler o que veio" desalinha a predicao em silencio. O timeout e' o da
    sessao: o fio se perdeu, e quem chama re-faz o handshake (a seq nao
    avancou; o log ja' gravado e' idempotente)."""
    buf = b""
    deadline = time.monotonic() + timeout_s
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if chunk:
            buf += chunk
        elif time.monotonic() > deadline:
            raise ConnectionResetError(f"serial: {len(buf)}/{n} bytes no timeout")
    return buf


def _serial_handshake(ser, reset: bool = False, tentativas: int = MAGIC_RETRIES,
                      intervalo_s: float = MAGIC_INTERVAL_S) -> None:
    """MAGIC do host, MAGIC_ACK do firmware, lixo de boot fora do caminho.

    reset=True manda o RESET (a seq do firmware volta a 0): e' o handshake do
    INICIO de cada serve_serial() (nova usina). reset=False manda o RESUME: e' a
    reentrada apos um soluço, e a seq fica onde estava. O reset_input_buffer()
    descarta o que o ROM escreveu na CDC antes de nos (e o resto de uma sessao
    anterior). Sem resposta, reenvia o MAGIC: o firmware pode ter consumido um
    como parte de uma troca interrompida (desync) e ficado aguardando outro.
    Depois do MAGIC_ACK o fluxo e' so' protocolo.
    """
    magia = bytes([MAGIC_RESET if reset else MAGIC])
    for _ in range(tentativas):
        ser.reset_input_buffer()
        ser.write(magia)
        deadline = time.monotonic() + intervalo_s
        while time.monotonic() < deadline:
            if ser.read(1) == bytes([MAGIC]):
                return
    raise ConnectionResetError("serial: firmware nao respondeu o MAGIC")


def serve_serial(df: pd.DataFrame, porta: str, baud: int = 115200,
                 log_path: Path = paths.EVAL_LOG) -> pd.DataFrame:
    """O laco do servidor pela CDC do USB-Serial/JTAG (COM29), em LOTES.

    O log do firmware fica na UART0 (COM28). O host manda lotes de BATCH amostras
    numa escrita so' e recebe lotes de predicoes numa leitura so': um par de
    transferencias por lote em vez de duas por linha (o downlink deste elo tem
    soluços de ~2 s a cada ~30 mensagens pequenas; em lote, um soluço custa UM
    reenvio por ~64 linhas). O lote e' idempotente: reenviar o MESMO lote
    reprocessa as mesmas amostras. Quando a leitura exata estoura o timeout (5 s,
    contra 2 s do firmware - o cliente desiste primeiro), o handshake RESUME
    refaz a sessao e o host reenvia o lote pendente. Termina com um lote vazio
    (count=0), que o firmware trata como fim do dataset e re-arma para a proxima
    usina. O print e' a cada lote e a cada 100 linhas logadas.
    """
    import serial  # pyserial: import aqui para o caminho WiFi nao depender dele

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    logged: set[int] = set()
    n = len(df)
    base = 0

    # serial_for_url: nome comum de porta (ex.: COM29) cai na classe da
    # plataforma; URL (ex.: loop://, nos testes) abre o dispositivo virtual.
    ser = serial.serial_for_url(porta, baud, timeout=0.2, write_timeout=1.0)
    ser.dtr = False     # nao asserta DTR/RTS: adaptador nenhum pode resetar a placa
    ser.rts = False
    with ser:
        _serial_handshake(ser, reset=True)
        print(f"Serial aberta em {porta} (baud virtual em USB): sessao iniciada", flush=True)

        with open(log_path, "w") as eval_log:
            eval_log.write("sample,real,pred,err\n")
            while base < n:
                fim = min(base + BATCH, n)
                count = fim - base
                payload = b"".join(
                    sample_bytes(df.iloc[i]) for i in range(base, fim))
                lote = struct.pack("<BII", REQUEST_SAMPLE, base, count) + payload
                try:
                    ser.write(lote)
                    resposta = _serial_read_exact(ser, 9 + 4 * count)
                except (ConnectionResetError, serial.SerialException):
                    print(f"serial: leitura perdida no lote base={base} - "
                          "refazendo o handshake (RESUME)", flush=True)
                    _serial_handshake(ser)
                    continue          # reenvia o MESMO lote (base nao avancou)
                tipo, r_base, r_count = struct.unpack("<BII", resposta[:9])
                if tipo != PREDICTION or r_base != base or r_count != count:
                    print(f"serial: resposta fora do lote (tipo 0x{tipo:02x}, "
                          f"base {r_base}/{base}, count {r_count}/{count}) - "
                          "refazendo o handshake (RESUME)", flush=True)
                    _serial_handshake(ser)
                    continue          # reenvia o MESMO lote
                preds = struct.unpack(f"<{count}f", resposta[9:])
                for i, pred in enumerate(preds):
                    record = _log_prediction(df, base + i, pred, logged,
                                             records, eval_log)
                    if record is not None and (len(records) % 100) == 1:
                        print(_record_line(record), flush=True)
                base = fim
                print(f"lote base={base - count}..{base - 1} ok "
                      f"({len(records)}/{n} linhas)", flush=True)

        # Fim do dataset: lote vazio; o firmware re-arma e aguarda a proxima usina.
        ser.write(struct.pack("<BII", REQUEST_SAMPLE, 0, 0))
        print(f"fim do dataset: {n} linhas", flush=True)
    return pd.DataFrame(records, columns=["sample", "pred", "real", "err"])


def eval_metrics(df_eval: pd.DataFrame) -> dict:
    """R2/MAE/vies/amplitude do log do servidor — os numeros do painel de avaliacao.

    Linhas com real NaN (gaps artificiais, sentinela de faltante) ficam fora do
    calculo: quem mede o erro DENTRO dos gaps junta o log com o dataset original.
    """
    df_eval = df_eval.dropna(subset=["real", "pred"])
    np_real = df_eval["real"].to_numpy()
    np_pred = df_eval["pred"].to_numpy()
    ss_res = np.sum((np_pred - np_real) ** 2)
    ss_tot = ((np_real - np.mean(np_real)) ** 2).sum()
    abs_err = np.abs(np_pred - np_real)
    denom = np.abs(np_real)
    teto = df_eval["pred"].max()
    return {
        "r2": 1 - (ss_res / ss_tot) if ss_tot else float("nan"),
        # Calculado de pred-real: nao depende do sinal/definicao da coluna `err`.
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean((np_pred - np_real) ** 2))),
        "mape": float(np.nanmean(abs_err / np.where(denom == 0, np.nan, denom)) * 100),
        "p90": float(np.quantile(abs_err, 0.90)),
        "vies": float((np_pred - np_real).mean()),
        "teto": float(teto),
        "no_teto": int((df_eval["pred"] >= teto - 0.01).sum()),
        "amostras": len(df_eval),
    }
