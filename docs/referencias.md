# Referências

Levantadas em 2026-11-03 para a definição dos hiperparâmetros do MLP por
granularidade (1 d, 1 h e 5 min).

## Hiperparâmetros e treino

### [1] Bengio (2012)

Bengio, Y. **Practical Recommendations for Gradient-Based Training of Deep
Architectures.** In: Montavon, G.; Orr, G. B.; Müller, K.-R. (eds.)
*Neural Networks: Tricks of the Trade*, 2. ed. Lecture Notes in Computer
Science, vol. 7700. Springer, 2012. p. 437–478.

- **DOI:** [10.1007/978-3-642-35289-8_26](https://doi.org/10.1007/978-3-642-35289-8_26)
- **Link:** [arXiv:1206.5533](https://arxiv.org/abs/1206.5533)

### [2] Masters e Luschi (2018)

Masters, D.; Luschi, C. **Revisiting Small Batch Training for Deep Neural
Networks.** arXiv preprint, 2018.

- **DOI (arXiv):** [10.48550/arXiv.1804.07612](https://doi.org/10.48550/arXiv.1804.07612)
- **Link:** [arXiv:1804.07612](https://arxiv.org/abs/1804.07612)

### [3] Kingma e Ba (2015)

Kingma, D. P.; Ba, J. **Adam: A Method for Stochastic Optimization.**
International Conference on Learning Representations (ICLR), 2015.

- **DOI (arXiv):** [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
- **Link:** [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)

### [4] Prechelt (1998)

Prechelt, L. **Early Stopping — But When?** In: Orr, G. B.; Müller, K.-R.
(eds.) *Neural Networks: Tricks of the Trade*. Lecture Notes in Computer
Science, vol. 1524. Springer, 1998. p. 55–69.

- **DOI:** [10.1007/3-540-49430-8_3](https://doi.org/10.1007/3-540-49430-8_3)
- **Reimpressão (2012):** [10.1007/978-3-642-35289-8_5](https://doi.org/10.1007/978-3-642-35289-8_5)

### [5] Keras — EarlyStopping

Keras. **EarlyStopping callback.** Documentação oficial.

- **Link:** [https://keras.io/api/callbacks/early_stopping/](https://keras.io/api/callbacks/early_stopping/)
- **DOI:** não se aplica (documentação).

### [6] scikit-learn — HuberRegressor

scikit-learn. **sklearn.linear_model.HuberRegressor.** Documentação oficial
(`epsilon=1.35`, 95% de eficiência estatística).

- **Link:** [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor)
- **DOI:** não se aplica (documentação).

## Previsão solar e granularidade

### [7] Dhingra, Gruosso e Storti Gajani (2025)

Dhingra, S.; Gruosso, G.; Storti Gajani, G. **A benchmark study of optimizers
for short-term solar PV power forecasting using neural networks under
real-world constraints.** *Neural Computing and Applications*, v. 37, n. 29,
p. 23909–23939, 2025.

- **DOI:** [10.1007/s00521-025-11546-2](https://doi.org/10.1007/s00521-025-11546-2)
- **Link:** [Springer](https://link.springer.com/article/10.1007/s00521-025-11546-2)

### [8] Abdul Rahman et al. (2025)

Abdul Rahman, N. H.; Sulaiman, S. I.; Hairuddin, M. A.; Hussin, M. Z.;
Mat Saat, E. H. **Sliding Window-Based Deep Learning Approach for Solar Power
Forecasting in Malaysia Utility-Scale PV Systems.** *Journal of Renewable
Energy and Smart Grid Technology*, v. 21, n. 1, p. 1–12, 2025.

- **DOI:** [10.69650/rast.2026.263014](https://doi.org/10.69650/rast.2026.263014)
- **Link:** [RAST](https://ph01.tci-thaijo.org/index.php/RAST/article/view/263014)

### [9] Aman et al. (2025)

Aman, R.; Dilshod, M.; Ulmas, Z.; Kumar, A.; Rizwan, M. **A novel hybrid
GWO-Bi-LSTM-based metaheuristic framework for short-term solar photovoltaic
power forecasting.** *Journal of Renewable and Sustainable Energy*, v. 17,
n. 4, art. 046101, 2025.

- **DOI:** [10.1063/5.0264101](https://doi.org/10.1063/5.0264101)
- **Link:** [AIP](https://pubs.aip.org/aip/jrse/article-abstract/17/4/046101/3352181/A-novel-hybrid-GWO-Bi-LSTM-based-metaheuristic)

### [10] Huber (1964)

Huber, P. J. **Robust Estimation of a Location Parameter.** *The Annals of
Mathematical Statistics*, v. 35, n. 1, p. 73–101, 1964.

- **DOI:** [10.1214/aoms/1177703732](https://doi.org/10.1214/aoms/1177703732)

## Uso na definição dos hiperparâmetros

| Tema | Referências |
|---|---|
| batch size e generalização | [1] Bengio; [2] Masters e Luschi; [7] Dhingra et al. |
| learning rate / Adam | [3] Kingma e Ba; [7] Dhingra et al.; [9] Aman et al. |
| early stopping e paciência | [1] Bengio; [4] Prechelt; [5] Keras |
| Huber delta | [6] scikit-learn; [10] Huber |
| granularidade 5 min × 1 h × 1 d | [7] Dhingra et al.; [8] Abdul Rahman et al.; [9] Aman et al. |
