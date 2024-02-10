MMD-reguaralized f-divergence Wasserstein-2 particle flows
=========================

This repository provides the method

`MMD_reg_f_div_flow` (from the file `MMD_reg_fDiv_ParticleFlows_CUDA.py`)

used to produces the numerical experiments for the paper

[Wasserstein Gradient Flows for Moreau Envelopes of f-Divergences in Reproducing Kernel Hilbert Spaces](https://arxiv.org/abs/2402.04613) by [Sebastian Neumayer](https://scholar.google.com/citations?user=NKL-mLgAAAAJ&hl=en&oi=ao), [Viktor Stein](https://viktorajstein.github.io/), and [Gabriele Steidl](https://page.math.tu-berlin.de/~steidl/).
This code is written and maintained by [Viktor Stein](mailto:stein@math.tu-berlin.de). Any comments and feedback are welcome!

The required packages are
---------------------------
* torch
* scipy
* numpy
* pillow (if you want to generate a gif of the evolution of the flow)
* matplotlib

The other python files contain auxillary functions.

Scripts to exactly reproduce the figures in the preprint are soon to come. An example file is `AlphaComparison.py`.
