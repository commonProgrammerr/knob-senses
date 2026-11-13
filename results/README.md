# results

Saídas geradas pelos notebooks. Cada tabela é gravada duas vezes, em `.csv` (consumo por
código) e `.md` (leitura direta), para que uma pessoa ou um agente consiga conferir o
resultado sem abrir o `.ipynb`. Os gráficos ficam em `figuras/`. A pasta `docs/figuras/`,
de onde o TCC inclui as figuras, não é escrita pelos notebooks: a cópia é manual, feita
só quando a figura vai para o texto.

Artefatos não ficam aqui: modelos e transformações continuam em `dist/`, e os splits
continuam em `data/splits/`.

## Estrutura

| pasta | origem | conteúdo |
|---|---|---|
| `ensaio/` | `notebooks/model_teste.ipynb` | logs por inversor (`{inversor}_val.csv`, `model_eval.csv`), métricas por período, validação dos gaps e `resumo_ensaio.md` |
| `figuras/` | todos os notebooks | gráficos em PNG |
| `dataset/` | `dataset_prep.ipynb`, `dataset_analysis.ipynb` | `splits_resumo.csv`, `correlacao.csv` |
| `treino_rf/` | `treino_random_forest.ipynb` | `metricas_rf`, `varredura_rf`, `configs_rf`, `importancia_rf`, `cruzada_rf` |
| `treino_mlp/` | `treino_mlp.ipynb` | `metricas_mlp`, `baseline_mlp`, `cruzada_mlp`, `historico_{freq}` |

## Ensaio

`ensaio/resumo_ensaio.md` é o ponto de entrada: reúne a tabela de métricas por período,
os resumos fora/dentro dos gaps e a lista das figuras do ensaio.

- `{inversor}_val.csv`: uma linha por amostra inferida (`sample, real, pred, err,
  inferencia_us`); `inferencia_us` é o tempo medido no firmware ao redor de `runInference()`.
- `metricas_ensaio.csv`: total e por mês (MAE, RMSE, R², viés, MAPE, P90 e tempo de inferência).
- `resumo_gaps.csv`: por inversor, métricas fora dos gaps e MAE do soft-sensor,
  da interpolação linear e da persistência dentro dos gaps.
- `resumo_gaps_duracao.csv`: o mesmo dentro dos gaps, desagregado por duração do bloco.
- `validacao_gaps.csv`: linha a linha dentro dos gaps, com as duas baselines.

## Como regenerar

Cada notebook grava ao rodar as células de resultado; não há etapa separada. O ensaio é o
único que depende da placa e sobrescreve `ensaio/{inversor}_val.csv` a cada execução.
