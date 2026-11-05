# Referências

Levantadas em 2026-11-03 para a definição dos hiperparâmetros do MLP por
granularidade (1 d, 1 h e 5 min). Ampliadas em 2026-11-05 para cobrir o restante
do pipeline: preparação e qualidade dos dados, split temporal, calibração de
viés, transformação Box-Cox do alvo, comparação RF × MLP, engenharia de
features, TinyML e métricas de regressão.

As referências [11]–[46] foram levantadas via Consensus; [16] e [26] são os
clássicos citados diretamente. [47]–[50] são documentação das ferramentas.

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

## Validação temporal, vazamento e drift

### [11] Cerqueira et al. (2019)

Cerqueira, V.; Torgo, L.; Mozetič, I. **Evaluating time series forecasting
models: an empirical study on performance estimation methods.** *Machine
Learning*, 2019.

- **Link:** [Consensus](https://consensus.app/papers/details/2d7493f839e35467a2e51f7d216cc8c2/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [12] Albelali et al. (2025)

Albelali, S. et al. **Hidden Leaks in Time Series Forecasting: How Data Leakage
Affects LSTM Evaluation Across Configurations and Validation Strategies.**
arXiv preprint, 2025.

- **Link:** [Consensus](https://consensus.app/papers/details/150a8d1c3bbf54fc8573652b2df8568c/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [13] Kingphai et al. (2026)

Kingphai, K. et al. **Adaptive Cross-Validation Under Concept Drift for Time
Series Forecasting.** *Journal of Computer Science*, 2026.

- **Link:** [Consensus](https://consensus.app/papers/details/7abf8c39d94e5ccb993ea07b39557b61/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [14] Kebir et al. (2024)

Kebir, S. et al. **On Handling Concept Drift, Calibration and Explainability
in Non-Stationary Environments and Resources Limited Contexts.** 2024
(veículo não identificado no resultado).

- **Link:** [Consensus](https://consensus.app/papers/details/52d1aed139965eb7ad31d70082973c6d/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [15] D'Elia et al. (2022)

D'Elia, G. et al. **Influence of Concept Drift on Metrological Performance of
Low-Cost NO2 Sensors.** *IEEE Transactions on Instrumentation and Measurement*,
2022.

- **Link:** [Consensus](https://consensus.app/papers/details/4bb4cec0ff2452c6aa7d84cc48023cd6/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

## Transformação do alvo

### [16] Box e Cox (1964) — clássico

Box, G. E. P.; Cox, D. R. **An Analysis of Transformations.** *Journal of the
Royal Statistical Society: Series B (Methodological)*, v. 26, n. 2,
p. 211–252, 1964.

- **DOI:** [10.1111/j.2517-6161.1964.tb00553.x](https://doi.org/10.1111/j.2517-6161.1964.tb00553.x)

### [17] Atkinson et al. (2021)

Atkinson, A. C. et al. **The Box–Cox Transformation: Review and Extensions.**
*Statistical Science*, 2021.

- **Link:** [Consensus](https://consensus.app/papers/details/51e355e91b47587096a672a00e8ee198/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [18] Riani et al. (2022)

Riani, M. et al. **Automatic robust Box–Cox and extended Yeo–Johnson
transformations in regression.** *Statistical Methods & Applications*, 2022.

- **Link:** [Consensus](https://consensus.app/papers/details/76445a4977565de598253dcac3f9af72/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

## Modelos para potência solar

### [19] Tercha et al. (2024)

Tercha, W. et al. **Machine Learning-Based Forecasting of Temperature and Solar
Irradiance for Photovoltaic Systems.** *Energies*, 2024.

- **Link:** [Consensus](https://consensus.app/papers/details/7d07658f50db5fbca9394e5020571cd2/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [20] Nahar et al. (2025)

Nahar, A. et al. **Forecasting Solar Photovoltaic Power Generation: A Machine
Learning Time Series Model Approach.** *International Journal of Energy
Research*, 2025.

- **Link:** [Consensus](https://consensus.app/papers/details/9c9369181fe255139b17fbe4e21a00c9/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [21] Sharadga et al. (2020)

Sharadga, H. et al. **Time series forecasting of solar power generation for
large-scale photovoltaic plants.** *Renewable Energy*, 2020.

- **Link:** [Consensus](https://consensus.app/papers/details/81b002614d445df19495f62df92e5860/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [22] Sangrody et al. (2020)

Sangrody, H. et al. **Similarity-Based Models for Day-Ahead Solar PV Generation
Forecasting.** *IEEE Access*, 2020.

- **Link:** [Consensus](https://consensus.app/papers/details/c11dccfd153753b29e8466771d0c723b/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [23] Terkes et al. (2026)

Terkes, M. et al. **Comparative analysis of AI-based regression models for
solar power forecasting: Techniques, preprocessing, and evaluation.** *Energy
Reports*, 2026.

- **Link:** [Consensus](https://consensus.app/papers/details/fd76bcf35a5b5af8b88d759a3efb986c/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [24] Al-Dahidi et al. (2024)

Al-Dahidi, S. et al. **A Novel Dynamic/Adaptive K-Nearest Neighbor Model for the
Prediction of Solar Photovoltaic Systems' Performance.** *Results in
Engineering*, 2024.

- **Link:** [Consensus](https://consensus.app/papers/details/6fe195681d6857eba7ae85715b86b775/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [25] Mittal et al. (2022)

Mittal, A. et al. **A Review on forecasting the photovoltaic power Using
Machine Learning.** *Journal of Physics: Conference Series*, 2022.

- **Link:** [Consensus](https://consensus.app/papers/details/fc3780a7bba858be8252960ac5ab6307/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

## Random Forest, redes neurais e tuning

### [26] Breiman (2001) — clássico

Breiman, L. **Random Forests.** *Machine Learning*, v. 45, n. 1, p. 5–32, 2001.

- **DOI:** [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)

### [27] Grinsztajn et al. (2022)

Grinsztajn, L. et al. **Why do tree-based models still outperform deep learning
on typical tabular data?** *Advances in Neural Information Processing Systems
35 (NeurIPS)*, 2022.

- **Link:** [Consensus](https://consensus.app/papers/details/3bc887a119c25257af04c0a792ee2971/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [28] Uddin et al. (2024)

Uddin, S. et al. **Confirming the statistically significant superiority of
tree-based machine learning algorithms over their counterparts for tabular
data.** *PLOS ONE*, 2024.

- **Link:** [Consensus](https://consensus.app/papers/details/dc1341adcf2455398d25c6c2ffdf59aa/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [29] Probst et al. (2018)

Probst, P. et al. **Hyperparameters and tuning strategies for random forest.**
*Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 2018.

- **Link:** [Consensus](https://consensus.app/papers/details/9af41b1693075bc5acbb29f5c17eabb8/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

## Calibração de viés

### [30] Cho et al. (2020)

Cho, D. et al. **Comparative Assessment of Various Machine Learning-Based Bias
Correction Methods for Numerical Weather Prediction Model Forecasts of Extreme
Air Temperatures in Urban Areas.** *Earth and Space Science*, 2020.

- **Link:** [Consensus](https://consensus.app/papers/details/4543de153dc45092857584a8dcf7ad78/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [31] Han et al. (2021)

Han, L. et al. **A Deep Learning Method for Bias Correction of ECMWF 24–240 h
Forecasts.** *Advances in Atmospheric Sciences*, 2021.

- **Link:** [Consensus](https://consensus.app/papers/details/0a9e8419575855b68e35e08f674d48bf/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

## Física da potência PV e engenharia de features

### [32] El Achouby et al. (2018)

El Achouby, H. et al. **New analytical approach for modelling effects of
temperature and irradiance on physical parameters of photovoltaic solar
module.** *Energy Conversion and Management*, 2018.

- **Link:** [Consensus](https://consensus.app/papers/details/2b2f2c1bb5125dd392980edd48f6c620/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [33] Krenzinger (2022)

Krenzinger, A. **Assessment on the variation of temperature coefficients of
photovoltaic modules with solar irradiance.** *Solar Energy*, 2022.

- **Link:** [Consensus](https://consensus.app/papers/details/cbe89413cb4e5a6ca4f6995557b32069/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [34] Naghapushanam et al. (2026)

Naghapushanam, M. et al. **Physics-informed voting ensemble for solar power
generation forecasting: integrating domain knowledge with machine learning.**
*Energy Informatics*, 2026.

- **Link:** [Consensus](https://consensus.app/papers/details/afcfdae083a45c6185b2e75fe5122dd8/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [35] Tao et al. (2024)

Tao, K. et al. **Operational day-ahead photovoltaic power forecasting based on
transformer variant.** *Applied Energy*, 2024.

- **Link:** [Consensus](https://consensus.app/papers/details/640423e5fd755fb08d15294ee15733ff/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

## TinyML e deploy embarcado

### [36] Sanchez-Iborra e Skarmeta (2020)

Sanchez-Iborra, R.; Skarmeta, A. F. **TinyML-Enabled Frugal Smart Objects:
Challenges and Opportunities.** *IEEE Circuits and Systems Magazine*, 2020.

- **Link:** [Consensus](https://consensus.app/papers/details/8d1a0e35c207529f91e6f357f2d656f8/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [37] Tabanelli et al. (2022)

Tabanelli, E. et al. **Optimizing Random Forest-Based Inference on RISC-V MCUs
at the Extreme Edge.** *IEEE Transactions on Computer-Aided Design of
Integrated Circuits and Systems*, 2022.

- **Link:** [Consensus](https://consensus.app/papers/details/34460aafdb285295b7d276073d7ddbd4/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [38] Lin et al. (2024)

Lin, J. et al. **Tiny Machine Learning: Progress and Futures [Feature].** *IEEE
Circuits and Systems Magazine*, 2024.

- **Link:** [Consensus](https://consensus.app/papers/details/570fbf3116cd5e01826cdd423b26fe47/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

## Qualidade de dados, clipping e soiling

### [39] Severiano et al. (2026)

Severiano, B. M. et al. **A robust rule-based method for detecting and
classifying underperformance in photovoltaic systems using inverter data.**
*Solar Energy*, 2026.

- **Link:** [Consensus](https://consensus.app/papers/details/3e06be4964f65c809159460a37dca57b/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [40] Balfour et al. (2021)

Balfour, J. et al. **Masking of photovoltaic system performance problems by
inverter clipping and other design and operational practices.** *Renewable and
Sustainable Energy Reviews*, 2021.

- **Link:** [Consensus](https://consensus.app/papers/details/8d8ebb8f10735856aa25c4686ea02d8a/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [41] Hara et al. (2021)

Hara, S. et al. **Exclusion of PV Inverter Measurement Data Affected by Power
Clipping and Detection of String Disconnection.** *2021 IEEE 48th Photovoltaic
Specialists Conference (PVSC)*, 2021.

- **Link:** [Consensus](https://consensus.app/papers/details/2b95cbf1172a571691325d7f1bf6bd3b/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [42] Said et al. (2026)

Said, S. et al. **Model for predicting photovoltaic soiling loss and
degradation of PV electrical performance using optimized machine learning.**
*Results in Engineering*, 2026.

- **Link:** [Consensus](https://consensus.app/papers/details/9bc049ad4a0058b88bd79d3c9724bd12/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [43] Tahir et al. (2025)

Tahir, M. et al. **Evaluating soiling effects to optimize solar photovoltaic
performance using machine learning algorithms.** *Energy Conversion and
Management: X*, 2025.

- **Link:** [Consensus](https://consensus.app/papers/details/5ccc36604859575ea97e6827017411c7/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

## Métricas de regressão

### [44] Chicco et al. (2021)

Chicco, D. et al. **The coefficient of determination R-squared is more
informative than SMAPE, MAE, MAPE, MSE and RMSE in regression analysis
evaluation.** *PeerJ Computer Science*, 2021.

- **Link:** [Consensus](https://consensus.app/papers/details/95adc7c5a9c35254b22f36bea2215497/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [45] Hodson (2022)

Hodson, T. O. **Root-mean-square error (RMSE) or mean absolute error (MAE):
when to use them or not.** *Geoscientific Model Development*, 2022.

- **Link:** [Consensus](https://consensus.app/papers/details/8bf35ca5be3f5b9db62b69d2dc3e33b8/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

### [46] Solorzano e Plevris (2022)

Solorzano, G.; Plevris, V. **Investigation of performance metrics in regression
analysis and machine learning-based prediction models.** *8th European Congress
on Computational Methods in Applied Sciences and Engineering (ECCOMAS)*, 2022.

- **Link:** [Consensus](https://consensus.app/papers/details/dde2e74e5c8a57d382d7d71f6739a466/?utm_source=claude_code)
- **DOI:** não disponível no resultado do Consensus.

## Ferramentas e documentação

### [47] emlearn

emlearn. **Embedded Machine Learning for microcontrollers.** Documentação
oficial (geração da lib C do Random Forest).

- **Link:** [https://emlearn.readthedocs.io/](https://emlearn.readthedocs.io/)
- **DOI:** não se aplica (documentação).

### [48] TensorFlow Lite for Microcontrollers

Google. **TensorFlow Lite for Microcontrollers.** Documentação oficial
(conversão e inferência do MLP no microcontrolador).

- **Link:** [https://www.tensorflow.org/lite/microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- **DOI:** não se aplica (documentação).

### [49] Edge Impulse

Edge Impulse. **Documentation.** Plataforma usada no caminho do MLP embarcado.

- **Link:** [https://docs.edgeimpulse.com/](https://docs.edgeimpulse.com/)
- **DOI:** não se aplica (documentação).

### [50] scikit-learn — RandomForestRegressor

scikit-learn. **sklearn.ensemble.RandomForestRegressor.** Documentação oficial
(implementação usada no treino e na varredura do RF).

- **Link:** [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- **DOI:** não se aplica (documentação).

## Uso na definição dos hiperparâmetros

| Tema | Referências |
|---|---|
| batch size e generalização | [1] Bengio; [2] Masters e Luschi; [7] Dhingra et al. |
| learning rate / Adam | [3] Kingma e Ba; [7] Dhingra et al.; [9] Aman et al. |
| early stopping e paciência | [1] Bengio; [4] Prechelt; [5] Keras |
| Huber delta | [6] scikit-learn; [10] Huber |
| granularidade 5 min × 1 h × 1 d | [7] Dhingra et al.; [8] Abdul Rahman et al.; [9] Aman et al. |

## Uso no pipeline e na comparação de modelos

| Decisão do projeto | Referências |
|---|---|
| split temporal `train`/`calibration`/`test`, sem embaralhar | [11] Cerqueira et al.; [12] Albelali et al.; [13] Kingphai et al. |
| leakage de normalização/transformação e avaliação honesta | [12] Albelali et al.; [11] Cerqueira et al. |
| drift de eficiência e calibração em janela recente | [14] Kebir et al.; [15] D'Elia et al.; [30] Cho et al.; [31] Han et al. |
| Box-Cox do alvo e inversa para kW | [16] Box e Cox; [17] Atkinson et al.; [18] Riani et al. |
| RF × MLP em dados tabulares | [26] Breiman; [27] Grinsztajn et al.; [28] Uddin e Lu |
| tuning do Random Forest (árvores/folha) | [29] Probst et al.; [50] scikit-learn |
| previsão de potência PV com irradiância/temperatura | [19] Tercha et al.; [20] Nahar et al.; [21] Sharadga et al.; [23] Terkes et al.; [25] Mittal et al. |
| resolução temporal (5 min × 1 h) | [21] Sharadga et al.; [22] Sangrody et al.; [7] Dhingra et al. |
| modelo físico e features de geometria solar | [32] El Achouby et al.; [33] Krenzinger; [34] Naghapushanam et al.; [35] Tao et al. |
| filtros de apagão, clipping e baixa irradiância | [39] Severiano et al.; [40] Balfour et al.; [41] Hara et al. |
| perda de eficiência por soiling/degradação | [42] Said et al.; [43] Tahir et al. |
| deploy em microcontrolador (emlearn/TFLite/Edge Impulse) | [36] Sanchez-Iborra e Skarmeta; [37] Tabanelli et al.; [38] Lin et al.; [47]–[49] |
| métricas MAE/RMSE/R² e viés | [44] Chicco et al.; [45] Hodson; [46] Solorzano e Plevris |
