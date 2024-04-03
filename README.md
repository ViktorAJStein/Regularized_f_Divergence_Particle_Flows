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
@unpublished{NSSR24,
 author = {Neumayer, Sebastian and Stein, Viktor and Steidl, Gabriele and Rux, Nicolaj},
 title = {Wasserstein Gradient Flows for {M}oreau Envelopes of $f$-Divergences in Reproducing Kernel {H}ilbert Spaces},
 note = {ArXiv preprint},
 volume = {arXiv:2402.04613},
 year = {2024},
 month = {Feb},
 url = {https://arxiv.org/abs/2402.04613},
 doi = {10.48550/arXiv.2402.04613}
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
2. Supported kernels
3. Supported $f$-divergences / entropy functions
4. Supported targets

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

Usually code is also compatible with some later or earlier versions of those packages.


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

We also implemented the following two "$W_2$-metrizing kernels", which metrize the Wasserstein-2 distance on $\mathcal P_2(\mathbb R^d)$, detailed in Example 4 of [Modeste, Dombry: "Characterization of translation invariant MMD on R d and connections with Wasserstein distances"](https://hal.science/hal-03855093).
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
Kullback-Leibler     | `tsallis`, \$\alpha = 1\$  | \$x \ln(x) - x + 1\$ for \$x > 0\$.
Tsallis-$`\alpha`$   | `tsallis`                  | \$\frac{1}{\alpha - 1} \left( x^{\alpha} - \alpha x + \alpha - 1 \right)\$
Jeffreys             | `jeffreys`                 | \$(x - 1) \ln(x)\$ for \$x > 0\$
$`\chi^{\alpha}`$    | `chi`                      | \$\| x - 1 \|^{\alpha}\$

Below we list some other implemented entropy functions with finite recession constant. For even more entropy functions we refer to table 1 in the above mentioned preprint.

Entropy              | Name             | Expression $f(x)$ for $x \ge 0$
---------------------| -----------------| ----------------------------------------------
Burg                 | `reverse_kl`     | $x - 1 - \ln(x)$ for $x > 0$
Jensen-Shannon       | `jensen_shannon` | $\log(x) - (x + 1) \ln\left(\frac{x+1}{2}\right)$ for $x > 0$
total variation      | `tv`             | $\| x - 1 \|$
Matusita             | `matusita`       | $\|1 - x^{\alpha} \|^{\frac{1}{\alpha}}$
Kafka                | `kafka`          | $\|1 - x \|^{\frac{1}{\alpha}} (1 + x)^{\frac{\alpha - 1}{\alpha}}$
Marton               | `marton`         | $\max(1 - x, 0)^2$
perimeter            | `perimeter`      | $\frac{\text{sign}(\alpha)}{1 - \alpha}\left( (x^{\frac{1}{\alpha}} + 1)^{\alpha} - 2^{\alpha - 1}(x + 1)\right)$

Supported targets
---------------------------
* `two_lines`: the two parabolas in the gif at the top
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

Speed
---------------------------
I am still working on improving the speed of this script, the bottleneck being the L-BFGS-B on the CPU.
Currently, running the simulation for 50000 steps (exact parameters: tsallis-divergence, alpha=3, lambd=1.0, tau=0.001, kernel = IMQ, sigma = 0.5, N = 900, target_name = bananas) takes less than 12 minutes on a CUDA 7.5 GPU with 12 GB of RAM.
