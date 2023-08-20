# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:38:40 2023

@author: Viktor Stein

Implements a relaxed Tsallis-divergence particle flow inspired by the KALE paper

Goal: use very small lambd to simulate approximate Tsallis flow
"""
import numpy as np
import matplotlib.pyplot as plt
# import cvxpy as cp
import scipy as sp
# from scipy.spatial.distance import pdist, squareform
import os
from PIL import Image # for gif creation

def get_timestamp(file_name):
    return int(file_name.split('-')[-1].split('.')[0])

def create_gif(image_folder, output_gif):
    images = []
    for filename in sorted(os.listdir(image_folder), key=get_timestamp):
        if filename.endswith(".png"):
            img = Image.open(os.path.join(image_folder, filename))
            images.append(img)
            

    if images:
        images[0].save(
            output_gif,
            save_all=True,
            append_images=images[1:],
            duration=70,  # You can adjust the duration between frames (in milliseconds) here
            loop=0  # 0 means infinite loop, change it to the number of loops you want
        )
        print(f"GIF saved as {output_gif}")
    else:
        print("No PNG images found in the folder.")

def kernel_matrix(a, kernel):
    result = []
    for i in range(a.shape[1]):
      current_result = []
      for j in range(a.shape[1]):
        x1 = a[:, i]
        x2 = a[:, j]
        current_result.append(kernel(x1, x2))
    
      result.append(current_result)
    
    return np.array(result)

alpha = 20 # Tsallis parameter >= 1
sigma = .05 # kernel parameter
mode = 'primal' # decide whether to solve the primal or dual problem
N = 100 # number of samples
#lambd = 1000 # --> 0 means Tsallis flow, --> infty means MMD flow
lambd = .01
#lambd = 1/np.sqrt(N) # (see bottom of p. 15 of KALE paper)
step_size = np.round(.1 * lambd, 5) # step size for Euler scheme
max_time = 25 # maximal time horizon (i.e. simulate flow until T = max_time)
iterations = int(max_time / step_size) + 1 # max number of iterations


np.random.seed(0) # fix randomness
# mean and variance of prior and target distributions
m_p, m_t = np.array([0, -2.25]), np.array([-1, -1])
v_p, v_t = 1/2000*np.array([[1, 0], [0, 1]]), 1/200*np.array([[1, 0], [0, 1]])
prior = np.random.multivariate_normal(m_p, v_p, N)
# prior = np.array([(k, 1) for k in np.linspace(-2, 2, N)])

# multiple circles target
#TODO: introduce parameter controlling number of rings
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
def gauss(x, y, s = sigma):
    d = x - y
    return np.exp(- 1/(2 * s) * np.dot(d.T, d)) 

def modified_Gauss(x, y, s = sigma):
    d = x - y
    x_squared = np.multiply(x, x)
    y_squared = np.multiply(y, y)
    return np.exp(- 1/(2 * s) * np.dot(d.T, d)) + np.dot(x_squared, y_squared)

# derivative of Gaussian kernel    
def gauss_Der(X, Y, s = sigma):
    return - 1 / s * (X - Y) * gauss(X, Y, s) 

# derivative of modified Gaussian kernel
def mod_Gauss_Der(X, Y, s = sigma):
    X_squared = np.multiply(X, X)
    return - 1 / s * (X - Y) * gauss(X, Y, s) + 2 * np. multiply(X_squared, Y)

# not differentiable!
# def laplace(x, y, s = sigma):
#     return np.exp(- 1 / s * np.abs(x - y).sum(axis=-1))

# negative distance / Riesz / energy kernel
def riesz(x, y, s = 3/4, sigma = 1):
    return (x**(2*s) + y**(2*s) - (x - y) ** (2*s)).sum(axis=-1) / sigma

def riesz_Der(x, y, s = 3/4, sigma = 1):
    if s == 1/2:
        return x / np.sqrt(np.dot(x, x)) + y / np.sqrt(np.dot(y, y)) - (x - y) / np.sqrt(np.dot(x - y, x - y))
    else:
        return (2*s) * ( x**(2 * s - 1) + y**(2 * s - 1) - (x - y)**(2 * s - 1)) 

def inv_Multiquadric(x, y, s = sigma, b = 1/2):
    return ( ((x - y) ** 2).sum(axis=-1) + s )**(-b)

def inv_Multiquadric_Der(x, y, s = sigma, b = 1/2):
    return - b * (((x - y) ** 2).sum(axis=-1) + s)**(-b-1) * (x - y)
    
def thin_Plate_Spline(x, y):
    d = ((x - y) ** 2).sum(axis=-1)
    return d*np.log(np.sqrt(d)) 

def thin_Plate_Spline_Der(x,y):
    d = np.sqrt(((x - y) ** 2).sum(axis=-1))
    return 2*(x - y) * (np.log(d) + 1/2)

def reLU(x):
    return 1/2*(x + np.abs(x))

# Tsallis and KL divergence generators and their derivatives
def tsallis_Generator(x, alpha):
    return np.choose(x >= 0, [np.inf, (x**alpha - alpha*x + alpha - 1)/(alpha - 1)])

def tsallis_Generator_Der(x, alpha):
    return np.choose(x >= 0, [np.inf, alpha / (alpha - 1) * ( x**(alpha - 1) - 1)])

def kl_Generator(x):
    return np.choose(x >= 0, [np.inf,np.choose(x > 0, [1, x*np.log(x)-x+1])])

def kl_Generator_Der(x):
    return np.choose(x > 0, [np.inf, np.log(x)])

# the Tsallis entropy function f_alpha
def entrop(x, alpha):
    if alpha != 1:
        return tsallis_Generator(x,alpha)
    else:
       return kl_Generator(x)

# derivative of f_alpha, f_alpha'    
def entrop_der(x, alpha):
    if alpha != 1:
        return tsallis_Generator_Der(x, alpha)
    else:
        return kl_Generator_Der(x)
    
# the conjugate f_alpha* of the entropy function f_alpha
def conj(x, alpha):
    if alpha != 1:
        return reLU((alpha - 1)/alpha * x + 1)**(alpha / (alpha - 1)) - 1
    else:
        return np.exp(x) - 1

# the derivative of f_alpha*, (f_alpha^*)*    
def conj_der(x, alpha):
    if alpha != 1:
        return reLU( (alpha - 1)/alpha * x + 1)**(1/(alpha - 1))
    else:
        return np.exp(x)

    
kern = inv_Multiquadric
kern_der = inv_Multiquadric_Der

kernel = kern.__name__
folder_name = f"alpha={alpha},lambd={lambd},tau={step_size},{kernel},{sigma}"
# Create the new folder in the current directory
try:
    os.mkdir(folder_name)
    print(f"Folder '{folder_name}' created successfully.")
except FileExistsError:
    print(f"Folder '{folder_name}' already exists.")
except Exception as e:
    print(f"An error occurred: {e}.")
    
## TODO: this should be a function
# now start Tsallis-KALE particle descent
KALE_value = np.zeros(iterations) 

for n in range(iterations):
    # concatenate sample of target and positions of particles in step n
    Z = np.append(X, Y, axis=0)
    M = len(Z)
    
    ## TODO: make this so we can use any kernel!!
    ## Compute gram matrix of the kernel with samples Z
    # pairwise_dists = squareform(pdist(Z, 'euclidean'))
    # K = np.exp(- pairwise_dists ** 2 / sigma)
    
    # K = kernel_matrix(Z, kern)
        
    K = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            K[i, j] = kern(Z[i,:], Z[j,:])

    
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
        else:
            warm_start = np.concatenate((-1/(lambd*N) * np.ones(N), np.zeros(N)))
            prob = sp.optimize.minimize(prim_objective, warm_start, jac=prim_jacobian, method='l-bfgs-b')
        beta = prob.x
        # beta[:N] = - 1/(lambd * N) * np.ones(N)
        
        h_star_grad = np.zeros((N, 2))
        for k in range(N):
            # TODO: vectorize the following line.
            temp = [beta[j] * kern_der(Y[k], Z[j]) for j in range(M)]
            # temp = [beta[j] * - 1/sigma * (Y[k] - Z[j]) * K[k+N, j] for j in range(M)]
            h_star_grad[k,:] =  np.sum(temp, axis=0)
 
        # gradient update
        # when vectorized, gives UFuncTypeError: Cannot cast ufunc 'subtract' output from dtype('float64') to dtype('int32') with casting rule 'same_kind'
        
        Y = Y - step_size * (1 + lambd) * h_star_grad
        
        KALE_value[n] = - (1 + lambd) * prim_objective(beta)
    
    if mode == 'dual':
        if n > 1: # warm start
            prob = sp.optimize.minimize(dual_objective, q, jac=dual_jacobian, method='l-bfgs-b', tol=1e-9)
        else:
            warm_start = np.ones(N)
            prob = sp.optimize.minimize(dual_objective, warm_start, jac=dual_jacobian, method='l-bfgs-b')
        q = prob.x
        
        
        if np.any(q < 0):
            print('Error: q is negative')
            break

        # gradient update        
        h_star_grad = np.zeros((N, 2))
        for k in range(N):
            # TODO: vectorize the following line.
            temp = [kern_der(Y[j], Y[k], sigma) - q[j] * kern_der(X[j], Y[k], sigma) for j in range(N)]
            h_star_grad[k,:] = 1/(lambd * N) * np.sum(temp, axis=0)

        Y = Y - step_size * (1 + lambd) * h_star_grad
        KALE_value[n] = (1 + lambd) * dual_objective(beta)

    # gradient of optimal function h_n^* evaluated at Y_n^(k)
    
    
    # plot the particles ten times per unit time interval
    if not n % int(1/(10*step_size)):
        Y_1, Y_2 = Y.T
        time = round(n*step_size, 1)
        plt.figure()
        plt.plot(Y_2, Y_1, 'x', label = 'Particles at \n' + fr'$t =${time}')
        plt.plot(target_y, target_x, '.', label = 'target') 
        # plt.plot(prior.T[1], prior.T[0], 'x', label = 'prior')
        plt.legend(loc='center left', frameon = False)
        plt.title('Tsallis-KALE particle flow solved via the ' + mode + ' problem, \n' + fr'$\alpha$ = {alpha}, $\lambda$ = {lambd}, $\tau$ = {step_size}, $N$ = {N},' + kernel + fr' $\sigma^2$ = {sigma}')
        #plt.ylim([-2.5, 2.5])
        #plt.xlim([-10, 2.5])
        plt.gca().set_aspect('equal')
        time_stamp = int(time*10)
        plt.savefig(folder_name + f'/T{alpha}_KALE_flow,lambd={lambd},tau={step_size}-{time_stamp}.png', dpi=300)
        plt.close()
plt.show() # show final result
plt.plot(range(iterations), KALE_value)
plt.title('Tsallis-KALE objective value (Problem solved via the ' + mode + ' problem, \n' + fr'$\alpha$ = {alpha}, $\lambda$ = {lambd}, $\tau$ = {step_size}, $N$ = {N},' + kernel + fr' $\sigma^2$ = {sigma}')

plt.show()
output_name = f'T{alpha}-KALE_flow,lambd={lambd},tau={step_size},{kernel}{sigma}.gif'    
create_gif(folder_name, output_name)


