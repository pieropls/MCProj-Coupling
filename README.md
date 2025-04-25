# Coupling

- **Project Title**: *Coupling 2 probability distributions with high probability of equality*
- **Course**: Simulation and Monte Carlo Methods (2nd Semester, ENSAE Paris)
- **Authors**: Piero PELOSI, Omar EL MAMOUNE, Sarakpy NY

## Project Overview

This project explores how to construct a **joint distribution** $(X,Y)$ for two given probability distributions $p$ and $q$ such that:
   - $X \sim p$, $Y \sim q$
   - $\mathbb{P}(X = Y) > 0$

Our work is based on the ideas from the paper [*The Coupled Rejection Sampler*](https://arxiv.org/abs/2201.09585) by Adrien Corenflos and Simo Särkkä (2022). We compare the proposed approach (the **Coupled Rejection Sampler** and its **Ensemble** variant) with a simpler but classical approach, **Thorisson’s algorithm** (Appendix I in the paper).

We also investigate the computational properties (e.g., running time, variance of the algorithm’s run time) in different scenarios:
- Coupling multivariate Gaussian distributions (including the case of different covariance matrices).
- Varying dimension.
- Considering potential random-walk Metropolis experiments (Section 5.3 of the paper).

The ultimate goal is to see how these coupling algorithms perform in practice and to gain insight into the pros and cons of each method.

<div align="center">
  <img src="pseudo-code/Cover.jpeg" alt="MC_project" width="400">
</div>

<p align="center">
  <sub>Cover generated with assistance from OpenAI's DALL·E.</sub>
</p>

## Repository structure

```
Coupling/
│
├── Coupling-code.ipynb                    # Final Jupyter notebook
│
├── project/
│   ├── Coupling-final-presentation.pdf    # Final slide deck
│   ├── Coupling.pdf                       # Main paper on The Coupled Rejection Sampler
│   └── Task.pdf                           # Task assignment
│
├── pseudo-code/                           # Visual pseudocode
│   ├── Cover.jpeg                         # AI-generated project cover image
│   ├── EnsembleRejectionCoupling.algo.png
│   ├── ReflectionMaximalCoupling.png
│   ├── RejectionCoupling.algo.png
│   └── Thorisson.algo.png
│
├── utils/                                 # Python utility scripts
│   ├── random_walk.py
│   ├── rejection_coupling_plot.py
│   └── tests.py
```

## Credits, course description & references

- **Course Instructor**: [Nicolas Chopin](https://nchopin.github.io)
- **Tutorials & Project Supervision**: Yvann Le Fay

The **Simulation and Monte Carlo Methods** course at ENSAE Paris focuses on techniques for **stochastic simulation** and their application to **statistics, machine learning, finance, and other domains**. Topics covered:
   - Monte Carlo integration and variance reduction techniques (e.g., antithetic variables, control variates).
   - Markov Chain Monte Carlo (MCMC) methods for Bayesian inference.
   - Importance Sampling and Quasi-Monte Carlo methods.
   - Applications to statistical inference, optimization, and financial engineering.
   - Implementation and analysis of simulation algorithms for high-dimensional problems.

The **literature available** for this project:
   - Corenflos, A. and Särkkä, S. (2022). *The Coupled Rejection Sampler*. [arXiv](https://arxiv.org/abs/2201.09585).
   - Thorisson, H. (2000). *Coupling, Stationarity, and Regeneration*. Springer.

## How to Use

1. Clone this repository to your local machine.
2. Navigate to the root directory and open the Jupyter notebook [`Coupling-code.ipynb`](Coupling-code.ipynb).
3. Run the notebook sequentially. Some cells may take up to 2 minutes depending on the algorithm and settings.
4. You can customize parameters such as dimension, target distribution, and number of samples directly within the notebook.
5. Results include visualizations of acceptance rates, coupling probabilities, and runtime comparisons.

## Final Grade

We received a final grade of **XX.XX/20**, which corresponds to a **X.X/4.0 GPA**.