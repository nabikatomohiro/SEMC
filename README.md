# Sequential Exchange Monte Carlo: Sampling Method for Bayesian Inference without Parameter Tuning

## Introduction
This repository contains the C++ code to reproduce the results of our paper: Tomohiro Nabika, Kenji Nagata, Shun Katakami, Masaichiro Mizumaki and Masato Okada, "Sequential Exchange Monte Carlo: Sampling Method for Bayesian Inference without Parameter Tuning." This repository has been created to allow readers to verify the effectiveness of the proposed Sequential Exchange Monte Carlo method. We define Replica Exchange Monte Carlo as REMC, Sequential Exchange Monte Carlo as SEMC and Transitional Markov Chain Monte Carlo as TMCMC. 
## Compiling and Running
We did numerical experiments in four problem settings: Sampling from artificial multimodal distribution, Spectral deconvolution(K=3), Spectral deconvolution(K=4) and Exhaustive search. Each source code is in the multimodal_samling/src, spectral_deconvolution/src, spectral_deconvolution_K_4/src and exhaustive_seach/src directory, respectively. 

To complie and run, go to the */src　directory, compile with "g++ file_name.cpp -O3 -fopenmp", and run ./a.out. Each .cpp file in */src folders are as follows:

- emc_*_base.cpp
  - For calculating baseline by REMC.
- emc_*_hyperparameter.cpp
  - For showing the Monte Calro parameter of REMC for section 4.2.1.
- emc_*.cpp
  - For executing REMC for section 4.2.2.
- semc_*_gamma.cpp
  - For executing REMC for section 4.2.2, 4.2.3, and SM3.
- semc_*_hyperparameter.cpp
  - For showing the Monte Calro parameter of SEMC for section 4.2.1.
- tmcmc_*.cpp
  - For executing TMCMC for section 4.2.3.
  
## Showing the Results
In the show directory, there are three .ipynb file Distribution_compare.ipynb, Freeenergy_compare.ipynb, and montecarlo_parameter.ipynb. 
Each file is for showing the result of distirubions distance, Free energy error, and Monte Calro parameter, respectively.
To show the result, execute in order from the top.

## Description for Other Files

- Eigen
  - C++ template library for linear algebra
- core/base_sampling.hpp
  - The base class for semc, remc and tmcmc
- core/base_emc.hpp
  - The class for remc
- core/base_semc.hpp
  - The class for semc
- core/base_tmcmc.hpp
  - The class for tmcmc
- x/src/y.hpp  ((x,y) = (multimodal_samling, multimodal_samling), (spectral_deconvolution, spectral_deconvolution), (spectral_deconvolution_K_4, spectral_deconvolution), (exhaustive_search, exhaustive_search))
  - The header file for each .cpp files


