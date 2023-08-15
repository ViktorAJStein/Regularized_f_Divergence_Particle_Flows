# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:38:40 2023

@author: Viktor Stein

Implements a relaxed Tsallis-divergence particle flow inspired by the KALE paper

Goal: use very small lambd to simulate approximate Tsallis flow
"""
import numpy as np
import matplotlib.pyplot as plt
#import cvxpy as cp
import scipy as sp
from scipy.spatial.distance import pdist, squareform


alpha = 1 # Tsallis parameter >= 1
max_iter = 100001 # max number of iterations
lambd = .01 # --> 0 means Tsallis flow, --> infty means MMD flow
step_size = .001 # step size for Euler scheme
sigma = .3 # kernel parameter
mode = 'primal' # solve the dual problem
N = 100 # number of samples
plot = True # decide whether to plot the particles

np.random.seed(0) # fix randomness
# mean and variance of prior and target distributions
m_p, m_t = np.array([0, -2.25]), np.array([-1, -1])
v_p, v_t = 1/20*np.array([[1, 0], [0, 2]]), 1/20*np.array([[2, 1], [1, 3]])
prior = np.random.multivariate_normal(m_p, v_p, N)
# prior = np.array([(k, 1) for k in np.linspace(-2, 2, N)])
# prior = np.array([(4, 0) for k in range(N)])

# multiple circles prior
Ntilde, r, _delta = int(N/2), 2, .25
X = np.c_[r * np.cos(np.linspace(0, 2 * np.pi, Ntilde + 1)), r * np.sin(np.linspace(0, 2 * np.pi, Ntilde + 1))][:-1]
for i in [1]: # for more circles, add more integers to this list
    X = np.r_[X, X[:Ntilde, :]-i*np.array([0, (2 + _delta) * r])]
target = X

# example with N = 1
# target = np.random.multivariate_normal(m_t, v_t, N)
# target = np.array([[0, 0]])
# prior = np.array([[1/2, 0]])
prior_x, prior_y = prior.T
target_x, target_y = target.T


Y = prior.copy() # samples of prior distribution
X = target # samples of target measure

# Gaussian kernel with width sigma
def gauss_kernel(x, y, s = sigma):
    d = x - y
    return np.exp(- 1/(2 * s) * np.dot(d.T, d))

## the following two kernels are taken from the KALE code (/kernel-wasserstein-flows/kernel_wasserstein_flows/kernels.py)
def laplace_kernel(x, y, sigma):
    return np.exp(-1 / sigma * np.abs(x - y).sum(axis=-1))

# todo: make this psd
def negative_distance_kernel(x, y, sigma):
    return  -((x - y) ** 2).sum(axis=-1) / sigma


def reLU(x):
    return 1/2*(x + np.abs(x))

# the Tsallis entropy function f_alpha
def entrop(x, alpha):
    if alpha != 1:
        return (x**alpha - alpha*x + alpha - 1)/(alpha - 1)
    else:
       return x*np.log(x) - x + 1

# derivative of f_alpha    
def entrop_der(x, alpha):
    if alpha != 1:
        return alpha/(alpha - 1) * (x**(alpha - 1) - 1)
    else:
        return np.log(x)
# the conjugate f* of the entropy function f
def conj(x, alpha):
    if alpha != 1:
        return 1/alpha * (reLU((alpha - 1) * x + 1))**(alpha / (alpha - 1)) - 1/alpha
    else:
        return np.exp(x) - 1

# the derivative of f*    
def conj_der(x, alpha):
    if alpha != 1:
        return alpha/ (alpha - 1) * (reLU( (alpha - 1) / alpha * x + 1))**(1/(alpha - 1))
    else:
        return np.exp(x)

# derivative of Gaussian kernel    
def kern_der(X, Y, sigma):
    return - 1 / sigma * (X - Y) * gauss_kernel(X, Y, sigma)
    

# now start Tsallis-KALE particle descent
for n in range(max_iter):
    # concatenate sample of target and positions of particles in step n
    Z = np.append(X, Y, axis=0)
    M = len(Z)
    
    # Compute gram matrix of the kernel with samples Z
    pairwise_dists = squareform(pdist(Z, 'euclidean'))
    K = np.exp(- pairwise_dists ** 2 / sigma ** 2)

    
    # this is minus the value of the KALE objective, if you multiply it by (1 + lambd)
    def prim_objective(b):
        p = K @ b        
        c1 = np.concatenate( (conj(p[:N], alpha), - p[N:]))
        c3 = b.T @ p
        return 1/N * np.sum(c1) + lambd/2 * c3
    
    # jacobian of the above ojective function
    def prim_jacobian(b):
        p = K @ b
        x = np.concatenate( (conj_der(p[:N], alpha), - np.ones(N)), axis=0)
        return 1/N * K @ x + lambd * p
    
    print(sp.optimize.check_grad( prim_objective, prim_jacobian, np.random.rand(2*N) ))
    # this fails, i.e. is equal to around .01 ...
    
    # dual objective is an N-dimensional function
    def dual_objective(q):
        convex_term = np.sum(entrop(q, alpha))
        tilde_q = np.concatenate((q, - np.ones(N)))
        quadratic_term = tilde_q.T @ K @ tilde_q
        return 1/N * convex_term + 1/(2 * lambd * N * N) * quadratic_term

    # jacobian of the above ojective function
    def dual_jacobian(q):
        convex_term = entrop_der(q, alpha)
        tilde_q = np.concatenate((q, - np.ones(N)))
        linear_term = K[:N, :] @ tilde_q
        return 1/N * convex_term + 1/(lambd * N * N) * linear_term

    # print(sp.optimize.check_grad( dual_objective, dual_jacobian, np.random.rand(N) ))
    
    if mode == 'primal':
        if n > 1: # warm start
            prob = sp.optimize.minimize(prim_objective, beta, jac=prim_jacobian, method='l-bfgs-b', tol=1e-9)
            # yields " Bad direction in the line search"
        else:
            warm_start = np.concatenate((-1/(lambd*N) * np.ones(N), np.zeros(N)))
            prob = sp.optimize.minimize(prim_objective, warm_start, jac=prim_jacobian, method='l-bfgs-b')
        beta = prob.x
        # beta[:N] = - 1/(lambd * N) * np.ones(N)
        
        h_star_grad = np.zeros((N, 2))
        for k in range(N):
            # TODO: vectorize the following line.
            temp = [beta[j] * (Y[k] - Z[j]) * K[k+N, j] for j in range(M)]
            h_star_grad[k,:] = - 1/sigma * np.sum(temp, axis=0)
     
        # plt.plot(beta)
        # plt.show()
        # gradient update
        # when vectorized, gives UFuncTypeError: Cannot cast ufunc 'subtract' output from dtype('float64') to dtype('int32') with casting rule 'same_kind'
        
        Y = Y - step_size * (1 + lambd) * h_star_grad
    
    if mode == 'dual':
        if n > 1: # warm start
            prob = sp.optimize.minimize(dual_objective, q, jac=dual_jacobian, method='l-bfgs-b', tol=1e-9)
        else:
            warm_start = np.zeros(N)
            prob = sp.optimize.minimize(dual_objective, warm_start, jac=dual_jacobian, method='l-bfgs-b')
        q = prob.x

        # gradient update        
        h_star_grad = np.zeros((N, 2))
        for k in range(N):
            # TODO: vectorize the following line.
            temp = [kern_der(Y[j], Y[k], sigma) - q[k] * kern_der(X[j], Y[k], sigma) for j in range(N)]
            h_star_grad[k,:] = 1/(lambd * N) * np.sum(temp, axis=0)
        Y = Y - step_size * (1 + lambd) * h_star_grad
        
    
    # K_psd = cp.psd_wrap(K) # make K numerically psd, even though it is theoretically psd    
    
    # # Solve primal convex M-dim problem for coefficients b
    # b = cp.Variable(M) # variable of length M
    # p = K_psd @ b
    # c1 = 1/N * cp.sum(p[N:])
    # c2 = - 1/N * cp.sum(conj(p[:N], alpha))
    # c3 = cp.quad_form(b, K_psd) # b.T @ K_psd @ b

    # prob = cp.Problem(cp.Minimize(- c1 - c2 + lambd/2 * c3))
    # if n > 0:
    #     prob.solve(warm_start=True, solver='ECOS') #verbose=True)
    # else:
    #     prob.solve()
    # beta = b.value
    
    # gradient of optimal function h_n^* evaluated at Y_n^(k)
    
    
    # plot every 100-th image
    if plot and not n % 100:
        Y_1, Y_2 = Y.T
        time = round(n*step_size, 1)
        plt.plot(Y_2, Y_1, '.', label = 'Particles at \n' + fr'$t =${time}')
        plt.plot(target_y, target_x, '.', label = 'target') 
        plt.plot(prior.T[1], prior.T[0], 'x', label = 'prior')
        plt.legend(loc='center left')
        plt.title('Tsallis-KALE particle flow solved via the ' + mode + ' problem with \n' + fr'$\alpha$ = {alpha}, $\lambda$ = {lambd}, $\tau$ = {step_size}, $N$ = {N} and Gauss kernel, $\sigma$ = {sigma}')
        plt.ylim([-2.5, 2.5])
        plt.xlim([-10, 2.5])
        plt.gca().set_aspect('equal')
        plt.savefig(f'T{alpha}-KALE_flow,lambd={lambd},tau={step_size}_{time}.png', dpi=300)
        plt.show()
    
