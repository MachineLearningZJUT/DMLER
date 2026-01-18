# Double/Debiased Machine Learning for Expectile Regression 

This repository contains the implementation code for the paper "Double/debiased machine learning for expectile regression". 

## Overview

This implementation provides a framework for debiased expectile regression using double/debiased machine learning techniques. The code reproduces the simulation results presented in Section 5 of the paper, specifically:

- **Figure 1**: Comparison of ER (Expectile Regression) and DMLER (Debiased Machine Learning Expectile Regression) estimators in high-dimensional systems for τ = 0.10, 0.50, and 0.90
- **Figure 3**: Comparison of ER and DMLER estimators in nonlinear systems for τ = 0.10, 0.50, and 0.90

## Repository Structure

```
.
├── biased_and_debiased.ipynb   
├── iterative_iv_plr.py           
├── requirements.txt             
├── Lasso_biased/                
├── Lasso_debiased/             
├── LGBM_biased/                  
├── LGBM_debiased/                
└── README.md                     
```

## File Descriptions

### `biased_and_debiased.ipynb`
The main experimental notebook that implements and compares biased and debiased estimators under two scenarios:
- **High-dimensional linear system**: Demonstrates the performance of ER vs DMLER in high-dimensional settings
- **Partially linear system**: Shows the comparison in nonlinear/partially linear models

This notebook generates the figures from Section 5 of the paper and provides visual comparisons of the estimation methods.

### `iterative_iv_plr.py`
Contains the core implementation of the DMLER framework, including:
- **DMLER estimator**: The debiased machine learning expectile regression model
- **Non-orthogonal ER**: The conventional (biased) expectile regression implementation
- **Expectile function**: Computation of the expectile function for the standard normal distribution
- **Asymmetric least squares**: Implementation of the asymmetric least squares method used in expectile regression

### `requirements.txt`
Lists all required Python packages with their specific versions to ensure reproducibility.

## Results

The experimental results are organized into four directories based on the machine learning method and estimation approach:

- **`Lasso_biased/`**: Contains figures and results for the biased expectile regression estimator using Lasso
- **`Lasso_debiased/`**: Contains figures and results for the debiased DMLER using Lasso
- **`LGBM_biased/`**: Contains figures and results for the biased expectile regression estimator using LightGBM
- **`LGBM_debiased/`**: Contains figures and results for the debiased DMLER using LightGBM

Each directory stores the comparison plots and numerical results for different expectile levels (τ = 0.10, 0.50, 0.90) in both high-dimensional linear and partially linear systems.

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

To reproduce the simulation results:

1. Open the Jupyter notebook:
```bash
jupyter notebook biased_and_debiased.ipynb
```

2. Run all cells in the notebook to:
   - Generate simulation data
   - Fit both biased (ER) and debiased (DMLER) estimators using Lasso and LightGBM
   - Produce comparison plots for different expectile levels (τ = 0.10, 0.50, 0.90)
   - Save results to the respective directories (Lasso_biased, Lasso_debiased, LGBM_biased, LGBM_debiased)

## Notes

- This is a demo version 
- The code is optimized for clarity and reproducibility
- Computation time may vary depending on the size of simulations and hardware specifications
