"""Os caminhos do repositorio, resolvidos do __file__ — valem de qualquer diretorio.

Antes isto saia de Path.cwd().parent, o que so' funcionava com o notebook aberto em
utils/; rodar de outro diretorio escrevia no lugar errado, em silencio.
"""

from pathlib import Path


def _find_root(start: Path) -> Path:
    """Sobe ate' achar a raiz do repo: pyproject.toml ao lado de utils/.

    Nao contar niveis (parents[2]) de proposito: o pacote ja' morou na raiz e hoje mora
    em models/src/, e um `parents[2]` fixo aponta para FORA do repo em um dos casos — o
    erro so' aparece no primeiro CSV lido, longe da causa.
    """
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() and (p / "utils").is_dir():
            return p
    raise RuntimeError(f"nao achei a raiz do repo (pyproject.toml + utils/) a partir de {start}")


ROOT = _find_root(Path(__file__).resolve().parent)

DATA = ROOT / "data"
RAW = DATA / "raw"                              # CSVs brutos por estacao-ano
PROCESSED = DATA / "processed"                  # saida do prep, 1 h por estacao
SPLITS = DATA / "splits"                       # saida do split (o que os treinos leem)
SCRATCH = DATA / "modelo_rf"                    # conferidor + residuo de build (gitignored)

LIB_RF = ROOT / "lib" / "knob-senses-rf"        # a entrega (gerada pelo notebook do RF)
LIB_RF_SRC = LIB_RF / "src"

LIB_KNN = ROOT / "lib" / "knob-senses-knn"      # a entrega do k-NN (gerada pelo notebook)
LIB_KNN_SRC = LIB_KNN / "src"

EVAL_LOG = ROOT / "utils" / "model_eval.csv"    # o log do servidor de teste

EXPORT_MLP = DATA / "modelo_exportado"          # SavedModel/zip/tflite do MLP
