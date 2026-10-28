"""A infraestrutura compartilhada do knob-senses.

Mapa dos modulos (importe o que precisar; este __init__ nao importa nada, para
`import common` nao puxar pandas, sklearn nem TensorFlow):

    contract   features/ordem, alvo, faixa fisica, limites e cortes do split
    paths      os caminhos do repo, resolvidos do __file__
    data       leitura/resumo dos CSVs, features de tempo, PCA_lag1, split, normalizacao
    metrics    evaluate() — MAE/RMSE/R2/vies, com clip
    templates  fill_template() — marcadores @@TOKEN@@ dos templates C
    verify     conferidor nativo (g++/xtensa) contra o sklearn (RF e k-NN)
    server     o protocolo do servidor de teste (model_teste.ipynb)
    plots      paleta, enfeitar e funcoes de grafico dos notebooks
"""
