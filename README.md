# Generalized Recursive Smooth Ambiguity Model

An implementation in Python of the model proposed in  
   Ju & Miao (2012). Ambiguity, Learning, and Asset Returns. *Econometrica*, 80(2), 559‑591.

Implementation by Eric André and Silvia Faroni, 2023.  
Available as open source under the terms of the MIT License.

---

Tested with
- python 3.10.13
- numpy 1.26.3
- pandas 2.1.4
- scipy 1.11.4
- matplotlib 3.8.0
- seaborn 0.12.2
- tqdm 4.65.0
  
---

Contents:
- `grsam.py`: main class definition.
- `01_Quadrature.ipynb`: test of the Gauss-Hermite quadrature with bivariate Normal distribution $\mathcal{N}(m, \Sigma)$ and Bayesian Normal prior over the mean vector $m \sim \mathcal{N}(\mu, \Lambda)$.
- `02_StaticPortfolios.ipynb`: compute static optimal portfolios with CRRA EU and KMM.