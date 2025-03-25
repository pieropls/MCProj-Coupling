# Coupling Two Probability Distributions

**Project Title**: *Coupling Two Probability Distributions with High Probability of Equality*.  
**Course**: Simulation and Monte Carlo Methods (2nd Semester, ENSAE Paris).  
**Authors**: Piero PELOSI, Omar EL MAMOUNE, Sarakpy NY.  

---

## Project Overview

This project explores how to construct a **joint distribution** $(X,Y)$ for two given probability distributions $p$ and $q$ such that:  
- $X \sim p$
- $Y \sim q$
- $\mathbb{P}(X = Y)$ is made as large as possible.  

Our work is based on the ideas from the paper [*The Coupled Rejection Sampler*](https://arxiv.org/abs/2201.09585) by Adrien Corenflos and Simo Särkkä (2022). We compare the proposed approach (the **Coupled Rejection Sampler** and its **Ensemble** variant) with a simpler but classical approach, **Thorisson’s algorithm** (Appendix I in the paper).  

We also investigate the computational properties (e.g., running time, variance of the algorithm’s run time) in different scenarios:  
- Coupling multivariate Gaussian distributions (including the case of different covariance matrices).  
- Varying dimension.  
- Considering potential random-walk Metropolis experiments (Section 5.3 of the paper).  

The ultimate goal is to see how these coupling algorithms perform in practice and to gain insight into the pros and cons of each method.

---

## Files in this Repository

1. **Coupling.pdf**  
   - Open-source article by Adrien Corenflos and Simo Särkkä.  
   - Available at [arXiv:2201.09585](https://arxiv.org/abs/2201.09585).  
   - Contains the theoretical foundations, algorithms, and results on coupled sampling methods.

2. **Coupling.ElMamoune.NY.Pelosi.ipynb**  
   - Our final Jupyter Notebook implementation.  
   - Includes: *To be completed*.  

3. **Cover.jpeg**  
   - The cover image generated for this project.  

---

## Credits & Course Description

- **Course Instructor**: [Nicolas Chopin](https://nchopin.github.io)  
- **Tutorials & Project Supervision**: Yvann Le Fay  
- **Institution**: ENSAE Paris  

The **Simulation and Monte Carlo Methods** course at ENSAE Paris focuses on techniques for **stochastic simulation** and their application to **statistics, machine learning, finance, and other domains**. Topics covered include:  
- Monte Carlo integration and variance reduction techniques (e.g., antithetic variables, control variates).  
- Markov Chain Monte Carlo (MCMC) methods for Bayesian inference.  
- Importance Sampling and Quasi-Monte Carlo methods.  
- Applications to statistical inference, optimization, and financial engineering.  
- Implementation and analysis of simulation algorithms for high-dimensional problems.  

---

## How to Use

1. **Clone this repository** to your local machine.  
2. Open the Jupyter Notebook `Coupling.ElMamoune.NY.Pelosi.ipynb`.  
3. Run each section in order:  
   - **Section 1**: *To be completed*.  
   - **Section 2**: *To be completed*.  
   - **Section 3**: *To be completed*.  
   - **Section 4**: *To be completed*.  
   - **Section 5**: *To be completed*.  
4. Adjust parameters (dimensions, distributions, number of samples) as desired.  
5. Observe the outputs: acceptance probabilities, run times, coupling probabilities, and other statistics.  

---

## References

- Corenflos, A. and Särkkä, S. (2022). *The Coupled Rejection Sampler*. [arXiv:2201.09585](https://arxiv.org/abs/2201.09585)  
- Thorisson, H. (2000). *Coupling, Stationarity, and Regeneration*. Springer.  

---

## Final Grade

We received a final grade of *to be completed*/20, equivalent to a *to be completed*/4.0 GPA.  

---

We hope you find this repository useful for understanding and experimenting with **coupled sampling** methods. Feel free to open an issue or submit a pull request if you have any questions, suggestions, or improvements!  
