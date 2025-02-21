![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

MMD-regularized $f$-divergence Wasserstein-2 particle flows
=========================

A python script to evaluate and plot the discretized Wasserstein-2 gradient flow starting at an empirical measure with respect to an Maximum-Mean-Discrepancy-regularized f-divergence functional, whose target is an empirical measure as well.

<p align="center">
  <img src="https://github.com/ViktorAJStein/Regularized_f_Divergence_Particle_Flows/blob/main/images/Bananas.gif" width="300" /> 
</p>

Overview
---------------------------
This repository provides the method

`MMD_reg_f_div_flow` (from the file [`MMD_reg_fDiv_ParticleFlows.py`](https://github.com/ViktorAJStein/Regularized_f_Divergence_Particle_Flows/blob/main/MMD_reg_fDiv_ParticleFlows.py))

used to produce the numerical experiments for the paper

[Wasserstein Gradient Flows for Moreau Envelopes of f-Divergences in Reproducing Kernel Hilbert Spaces](https://arxiv.org/abs/2402.04613) by [Sebastian Neumayer](https://scholar.google.com/citations?user=NKL-mLgAAAAJ&hl=en&oi=ao), [Viktor Stein](https://viktorajstein.github.io/), [Gabriele Steidl](https://page.math.tu-berlin.de/~steidl/) and [Nikolaj Rux](https://www.linkedin.com/in/nicolaj-rux-b14b44299/).

If you use this code please cite this preprint, preferably like this:
```
@unpublished{SNRS24,
 title={Wasserstein gradient flows for {M}oreau envelopes of $f$-divergences in reproducing kernel {H}ilbert spaces},
 author={Stein, Viktor and Neumayer, Sebastian and Rux, Nicolaj and Steidl, Gabriele},
 note = {ArXiv preprint},
 volume = {arXiv:2402.04613},
 year = {2024},
 month = {Feb},
 url = {https://arxiv.org/abs/2402.04613},
 doi = {10.1142/S0219530525500162}
 }
```

The other python files contain auxillary functions.
Scripts to exactly reproduce the figures in the preprint are soon to come. An example file is `AlphaComparison.py`.


Feedback / Contact
---------------------------
This code is written and maintained by [Viktor Stein](mailto:stein@math.tu-berlin.de). Any comments, feedback, questions and bug reports are welcome!
Alternatively you can use the [GitHub issue tracker](https://github.com/ViktorAJStein/Regularized_f_Divergence_Particle_Flows/issues).

Contents
---------------------------
1. Required packages
2. Options of the main method
3. Supported kernels
4. Supported $f$-divergences / entropy functions
5. Supported targets

Required packages
---------------------------
This script requires the following Python packages. We tested the code with Python 3.11.7 and the following package versions:

* torch 2.1.2
* scipy 1.12.0
* numpy 1.26.3
* pillow 10.2.0 (if you want to generate a gif of the evolution of the flow)
* matplotlib 3.8.2
* pot 0.9.3 (if you want to evaluate the exact Wasserstein-2 loss along the flow)
* sklearn.datasets 1.4.1.post1 (for more targets)
* https://github.com/gmgeorg/torchlambertw/

Usually code is also compatible with some later or earlier versions of those packages.

Options of the main method
---------------------------

Parameter | Type    | Explanation 
----------| --------|-------
a         | float   | divergence parameter 
s         | float   | kernel parameter > 0
N         | int     | number of prior particles
M         | int     | number of target particles
lambd     | float   | regularization parameter > 0
step_size | float   | step size for Euler forward discretization
max_time  | float   | maximal time horizon for simulation
plot      | boolean | decide whether to plot particles along the evolution
arrows    | boolean | decide whether to plot arrows at particles to show their gradients
timeline  | boolean | decide whether to plot timeline of functional value along the flow
kern      | function| kernel (see below)
primal    | bolean  | decide whether to solve the primal problem
dual      | bolean  | decide whether to solve the dual problem
div       | class entr_fnc | entropy function
target_name| string  | name of the target measure nu
verbose    | boolean | decide whether to print warnings and information
compute_W2 | boolean | decide whether to compute W2 dist of particles to target along flow
save_opts  | boolean | decide whether to save minimizers and gradients along the flow
st         | int      | random state for reproducibility
annealing  | boolean  | decide wether to use the annealing heuristic
annealing_factor | int | factor by which to divide lambda
tight      | boolean  |  decide whether to use the tight variational formulation
line_search| string   | step size choice for the exponetial GD for the tight formulation, either 'const', 'armijo', 'Polyak' or 'two_way'

Supported kernels
---------------------------
The following kernels all are radial and twice-differentiable, hence fulfilling all assumptions in the paper.
We denote the reLU by $(x)_+ := \max(x, 0)$ and the Euclidean norm by $\| \cdot \|$ and $s > 0$ the shape parameter.

Kernel                   | Name       | Expression $K(x, y) =$
-------------------------| -----------| ----------------------------------------------
inverse multiquadric     | `imq`      | $(s + \| x - y \|^2)^{-\frac{1}{2}}$ 
Gauss                    | `gauss`    | $\exp\left(- \frac{1}{2 s} \| x - y \|^2\right)$
Matérn-$`\frac{3}{2}`$   | `matern`   | $\left(1 + \frac{\sqrt{3} \| x - y \|}{s}\right) \exp\left(- \frac{\sqrt{3} \| x - y \|}{s}\right)$
Matérn-$`\frac{5}{2}`$   | `matern2`  | $\left(1 + \frac{\sqrt{5} \| x - y \|}{s} + \frac{5 \| x - y \|^2}{3 s^2} \right) \exp\left(- \frac{\sqrt{5} \| x - y \|}{s}\right)$
$B_{2\ell+1}$-Spline     | `compact`  | $(1 - \| x - y \|)_{+}^{s + 2}$
Another Spline           | `compact2` | $(1 - \| x - y \|)_{+}^{s + 3} \left( (s + 3) \| x - y \| + 1 \right)$
inverse log              | `inv_log`  | $\left(s + \ln(1 + \| x - y \|^2)\right)^{-\frac{1}{2}}$
inverse quadric          | `inv_quad` | $(1 + s \| x - y \|^2)^{-1}$
student t                | `student`  | $\frac{\Gamma\left(\frac{s + 1}{2}\right)}{\sqrt{s \pi} \Gamma\left(\frac{s}{2}\right)} \left(1 + \frac{1}{s} \| x - y \|^2\right)^{- \frac{s + 1}{2}}$

I also implemented the following two "$W_2$-metrizing kernels", which metrize the Wasserstein-2 distance on $\mathcal P_2(\mathbb R^d)$, detailed in Example 4 of [Modeste, Dombry: "Characterization of translation invariant MMD on R d and connections with Wasserstein distances"](https://hal.science/hal-03855093).
However, they are unbounded, not translation invariant (and one is not differentiable on the diagonal), so they do not fulfill our assumptions.
Kernel                   | Name       | Expression $K(x, y) =$
-------------------------| -----------| ----------------------------------------------
$W_2$-metrizing I        | `W2_1`     | $\exp\left(- \frac{1}{2 s} \| x - y \|^2\right) + \sum_{k = 1}^d \| x_k y_k \|$ 
$W_2$-metrizing II       | `W2_2`     | $\exp\left(- \frac{1}{2 s} \| x - y \|^2\right) + \sum_{k = 1}^d x_k^2 y_k^2$

Supported f-divergences / entropy functions
---------------------------
The following entropy functions each have an infinite recession constant if $\alpha > 1$.

Entropy              | Name                       | Expression \$f(x)\$ for \$x \ge 0\$
---------------------| ---------------------------| ----------------------------------------------
Kullback-Leibler     | `tsa`, \$\alpha = 1\$      | \$x \ln(x) - x + 1\$ for \$x > 0\$.
Tsallis-$`\alpha`$   | `tsa`                      | \$\frac{1}{\alpha - 1} \left( x^{\alpha} - \alpha x + \alpha - 1 \right)\$
Jeffreys             | `jeffreys`                 | \$(x - 1) \ln(x)\$ for \$x > 0\$
$`\chi^{\alpha}`$    | `chi_entr`                 | \$\| x - 1 \|^{\alpha}\$

Below we list some other implemented entropy functions with finite recession constant. For even more entropy functions we refer to table 1 in the above mentioned preprint.

Entropy              | Name                 | Expression $f(x)$ for $x \ge 0$
---------------------| ---------------------| ----------------------------------------------
Burg                 | `reverse_kl`         | $x - 1 - \ln(x)$ for $x > 0$
Jensen-Shannon       | `jensen_shannon`     | $\log(x) - (x + 1) \ln\left(\frac{x+1}{2}\right)$ for $x > 0$
total variation      | `tv`                 | $\| x - 1 \|$
Matusita             | `matusita`           | $\|1 - x^{\alpha} \|^{\frac{1}{\alpha}}$
Kafka                | `kafka`              | $\|1 - x \|^{\frac{1}{\alpha}} (1 + x)^{\frac{\alpha - 1}{\alpha}}$
Marton               | `marton`             | $\max(1 - x, 0)^2$
perimeter            | `per`                | $\frac{\text{sign}(\alpha)}{1 - \alpha}\left( (x^{\frac{1}{\alpha}} + 1)^{\alpha} - 2^{\alpha - 1}(x + 1)\right)$
equality indicator   | 'equality_indicator' | $\iota_{1}$
zero                 | 'zero'               | $0$
Lindsay              | 'lind'               | $\frac{(x - 1)^2}{a + (1 - a) x}$

Supported targets
---------------------------
* `bananas`: the two parabolas in the gif at the top
* `circles`: three circles
<p align="center">
  <img src="https://github.com/ViktorAJStein/Regularized_f_Divergence_Particle_Flows/blob/main/images/Circles.gif" width="500" /> 
</p>

* `cross`: four versions of Neals funnel arranged in a cross shape
<p align="center">
  <img src="https://github.com/ViktorAJStein/Regularized_f_Divergence_Particle_Flows/blob/main/images/Cross.gif" width="500" /> 
</p>

* `GMM`: two exactly equal Gaussians which have a symmetry axis at $y = - x$
<p align="center">
  <img src="https://github.com/ViktorAJStein/Regularized_f_Divergence_Particle_Flows/blob/main/images/GMM.gif" width="500" /> 
</p>

* `four_wells`: a sum of four Gaussians, which don't have a symmetry axis. The initial measure is initiated at one of the Gaussians.
<p align="center">
  <img src="https://github.com/ViktorAJStein/Regularized_f_Divergence_Particle_Flows/blob/main/images/four_wells.png" width="500" /> 
</p>

* `swiss_role_2d`:
<p align="center">
  <img src="https://github.com/ViktorAJStein/Regularized_f_Divergence_Particle_Flows/blob/main/images/swiss_roll_2d.gif" width="500" /> 
</p>

We also include some target measures from `sklearn.data`: `moons`, `annulus`
<p align="center">
  <img src="https://github.com/ViktorAJStein/Regularized_f_Divergence_Particle_Flows/blob/main/images/moons.gif" width="300" />  &emsp; &emsp; &emsp;
  <img src="https://github.com/ViktorAJStein/Regularized_f_Divergence_Particle_Flows/blob/main/images/annulus.gif" width="300" /> 
</p>

and the three-dimensional data sets `swiss_role_3d` and `s_curve`.
<p align="center">
  <img src="https://github.com/ViktorAJStein/Regularized_f_Divergence_Particle_Flows/blob/main/images/s_curve.gif" width="300" />  &emsp; &emsp; &emsp;
  <img src="https://github.com/ViktorAJStein/Regularized_f_Divergence_Particle_Flows/blob/main/images/swiss_roll_3d.gif" width="300" /> 
</p>

Legal Information & Credits
--------------------------
Copyright (c) 2024 Viktor Stein

This software was written by Viktor Stein. It was developed at the Institute of Mathematics, TU Berlin. The author acknowledges support by the German Research Foundation within the project VI screen.

This is free software. You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version. If not stated otherwise, this applies to all files contained in this package and its sub-directories.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
