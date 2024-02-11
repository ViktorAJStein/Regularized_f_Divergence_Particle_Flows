MMD-regularized f-divergence Wasserstein-2 particle flows
=========================

A python script to evaluate and plot the discretized Wasserstein-2 gradient flow starting at an empirical measure with respect to an MMD-regularized f-divergence functional, whose target is an empirical measure as well.

<p align="center">
  <img src="https://github.com/ViktorAJStein/Regularized_f_Divergence_Particle_Flows/blob/main/images/Bananas.gif" width="500" /> 
</p>

References
---------------------------
This repository provides the method

`MMD_reg_f_div_flow` (from the file `[MMD_reg_fDiv_ParticleFlows.py](https://github.com/ViktorAJStein/Regularized_f_Divergence_Particle_Flows/blob/main/MMD_reg_fDiv_ParticleFlows.py)`)

used to produce the numerical experiments for the paper

[Wasserstein Gradient Flows for Moreau Envelopes of f-Divergences in Reproducing Kernel Hilbert Spaces](https://arxiv.org/abs/2402.04613) by [Sebastian Neumayer](https://scholar.google.com/citations?user=NKL-mLgAAAAJ&hl=en&oi=ao), [Viktor Stein](https://viktorajstein.github.io/), and [Gabriele Steidl](https://page.math.tu-berlin.de/~steidl/).

If you use this code please cite this paper.

The other python files contain auxillary functions.
Scripts to exactly reproduce the figures in the preprint are soon to come. An example file is `AlphaComparison.py`.

The required packages are
---------------------------
* torch 2.1.2
* scipy 1.12.0
* numpy 1.26.3
* pillow 10.2.0 (if you want to generate a gif of the evolution of the flow)
* matplotlib 3.8.2
* pot 0.9.3 (if you want to evaluate the exact Wasserstein-2 loss along the flow)
* warnings


Feedback / Contact
---------------------------
This code is written and maintained by [Viktor Stein](mailto:stein@math.tu-berlin.de). Any comments, feedback, questions and bug reports are welcome!

Supported kernels
---------------------------
The following kernels all are radial and twice-differentiable, hence fulfilling all assumptions in the paper.

Kernel               | Name    | Expression $K(x, y) =$
--------------------:| --------| ----------------------------------------------
inverse multiquadric | `IMQ`   | $(\sigma + \| x - y \|_2^2)^{-\frac{1}{2}}$ 

* Gauss $K(x, y) = \exp\left(- \frac{1}{2 \sigma} \| x - y \|_2^2\right)$
* Matérn-$\frac{3}{2}$ (`Matern`) $K(x, y) = (1 + \frac{\sqrt{3} \| x - y \|_2}{\sigma} \exp\left(- \frac{\sqrt{3} \| x - y \|_2}{sigma}\right)
* Matérn-$\frac{5}{2}$ (`Matern2`) $K(x, y) = (1 + torch.sqrt(5*r) / sigma + 5*r/(3*sigma**2) ) * (- torch.sqrt(5*r) / sigma).exp()


Supported f-divergences / entropy functions
---------------------------
