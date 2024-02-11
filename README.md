MMD-regularized f-divergence Wasserstein-2 particle flows
=========================

A python script to evaluate and plot the discretized Wasserstein-2 gradient flow starting at an empirical measure with respect to an MMD-regularized f-divergence functional, whose target is an empirical measure as well.

<p align="center">
  <img src="https://github.com/ViktorAJStein/Regularized_f_Divergence_Particle_Flows/blob/main/images/Bananas.gif" width="500" /> 
</p>

References
---------------------------
This repository provides the method

`MMD_reg_f_div_flow` (from the file [`MMD_reg_fDiv_ParticleFlows.py`](https://github.com/ViktorAJStein/Regularized_f_Divergence_Particle_Flows/blob/main/MMD_reg_fDiv_ParticleFlows.py))

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
We denote the reLU by $(x)_+ := \max(x, 0)$ and the Euclidean norm by $\| \cdot \|$.

Kernel               | Name       | Expression $K(x, y) =$
---------------------| -----------| ----------------------------------------------
inverse multiquadric | `IMQ`      | $(\sigma + \| x - y \|^2)^{-\frac{1}{2}}$ 
Gauss                | `gauss`    | $\exp\left(- \frac{1}{2 \sigma} \| x - y \|^2\right)$
Matérn-$\frac{3}{2}$ | `Matern`   | $\left(1 + \frac{\sqrt{3} \| x - y \|}{\sigma}\right) \exp\left(- \frac{\sqrt{3} \| x - y \|}{\sigma}\right)$
Matérn-$\frac{5}{2}$ | `Matern2`  | $\left(1 + \frac{\sqrt{5} \| x - y \|}{\sigma} + \frac{5 \| x - y \|^2}{3 \sigma^2} \right) \exp\left(- \frac{\sqrt{5} \| x - y \|}{\sigma}\right)$
Spline               | `compact`  | $(1 - \| x - y \|)_{+}^{q + 2}$
Another Spline       | `compact2` | $(1 - \| x - y \|)_{+}^{q + 3} \left( (q + 3) \| x - y \| + 1 \right)$ 


Supported f-divergences / entropy functions
---------------------------
The following entropy functions each have an infinite recession constant if $\alpha > 1.

Entropy              | Name                     | Expression $f(x)$ for $x \ge 0$
---------------------| -------------------------| ----------------------------------------------
Kullback-Leibler     | `tsallis`, $\alpha = 1$  | $x \ln(x) - x + 1$ for $x > 0$.
Tsallis-$\alpha$     | `tsallis`                | $\frac{x^{\alpha} - \alpha x + \alpha - 1}{\alpha - 1}$
Jeffreys             | `jeffreys`               | $(x - 1) \ln(x)$ for $x > 0$
chi-$\alpha$         | `chi`                    | $| x - 1 |^{\alpha}$

Below we list some other implemented entropy functions with finite recession constant. For even more entropy functions we refer to table 1 in the above mentioned preprint.

Entropy              | Name             | Expression $f(x)$ for $x \ge 0$
---------------------| -----------------| ----------------------------------------------
Burg                 | `reverse_kl`     | $x - 1 - \ln(x)$ for $x > 0$
Jensen-Shannon       | `jensen_shannon` | $\log(x) - (x + 1) \ln\left(\frac{x+1}{2}\right)$ for $x > 0$
reverse Pearson      | `reverse_pearson`| $\frac{1}{x} - 1$ for $x > 0$
total variation      | `tv`             | $| x - 1 |$

Supported targets
---------------------------
* `two_lines`: the two parabolas in the gif at the top
*  `circles`: three circles
<p align="center">
  <img src="https://github.com/ViktorAJStein/Regularized_f_Divergence_Particle_Flows/blob/main/images/Circles.gif" width="500" /> 
</p>
* `cross`: four versions of Neals funnel arranged in a cross shape
<p align="center">
  <img src="https://github.com/ViktorAJStein/Regularized_f_Divergence_Particle_Flows/blob/main/images/Cross.gif" width="500" /> 
</p>
