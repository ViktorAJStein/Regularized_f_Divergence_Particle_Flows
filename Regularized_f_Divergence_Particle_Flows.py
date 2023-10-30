# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:38:40 2023

@author: Viktor Stein

Implements a RKHS-Moreau-envelope regularized f-divergence particle flow inspired by the KALE paper

"""
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import scipy as sp
from scipy.spatial.distance import pdist, squareform
import os
from PIL import Image # for gif creation
import time
import csv

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


np.random.seed(2) # fix randomness

# Gaussian kernel with width s
def gauss(x, y, s):
    d = x - y
    return np.exp(- 1/(2 * s) * np.dot(d.T, d)) 

# derivative of Gaussian kernel    
def gauss_der(X, Y, s):
    return - 1 / s * (X - Y) * gauss(X, Y, s) 


# # modified Gaussian kernel (metrizes Wasserstein-2)
# def mod_gauss(x, y = np.ones(10), s = sigma):
#     x_squared = np.multiply(x, x)
#     y_squared = np.multiply(y, y)
#     return np.dot(x_squared, y_squared) + gauss(x, y, s)


# # derivative of modified Gaussian kernel (is wrong??)
# def mod_gauss_der(x, y = np.ones(10), s = sigma):
#     x_squared = np.multiply(x, x)
#     return 2 * np.multiply(x_squared, y) + gauss_der(x, y, s) 


# negative distance / Riesz / energy kernel
def riesz(x, y, s):
    return LA.norm(x)**(2*s) + LA.norm(y)**(2*s) - LA.norm(x - y) ** (2*s)

# yields error for dividing by zero
def riesz_der(x, y, s):
    return 2 * s * LA.norm(y)**(2*s-2) * y - 2*s*LA.norm(x - y)**(2*s-2) * (x - y)

def inv_multiquadric(x, y, s = .05, b = 1/2):
    return ( ((x - y) ** 2).sum(axis=-1) + s )**(-b)

def inv_multiquadric_der(x, y, s = .05, b = 1/2):
    return - 2*b * (((x - y) ** 2).sum(axis=-1) + s)**(-b-1) * (x - y)

def reLU(x):
    return 1/2*(x + np.abs(x))

# f-divergence generators and their derivatives
def tsallis_generator(x, alpha):
    return np.choose(x >= 0, [np.inf, (x**alpha - alpha*x + alpha - 1)/(alpha - 1)])

def tsallis_generator_der(x, alpha):
    return np.choose(x >= 0, [np.inf, alpha / (alpha - 1) * ( x**(alpha - 1) - 1)])

def kl_generator(x):
    return np.choose(x >= 0, [np.inf,np.choose(x > 0, [1, x*np.log(x)-x+1])])

def kl_generator_der(x):
    return np.choose(x > 0, [np.inf, np.log(x)])

# the Tsallis entropy function f_alpha
def tsallis(x, alpha):
    if alpha != 1:
        return tsallis_generator(x,alpha)
    else:
       return kl_generator(x)

# derivative of f_alpha, f_alpha'    
def tsallis_der(x, alpha):
    if alpha != 1:
        return tsallis_generator_der(x, alpha)
    else:
        return kl_generator_der(x)
    
# the conjugate f_alpha* of the entropy function f_alpha
def tsallis_conj(x, alpha):
    if alpha != 1:
        return reLU((alpha - 1)/alpha * x + 1)**(alpha / (alpha - 1)) - 1
    else:
        return np.exp(x) - 1

# the derivative of f_alpha*, (f_alpha^*)'    
def tsallis_conj_der(x, alpha):
    if alpha != 1:
        return reLU( (alpha - 1)/alpha * x + 1)**(1/(alpha - 1))
    else:
        return np.exp(x)
    
def jeffreys(x):
    return np.choose(x > 0, [np.inf, (x - 1) * np.log(x)])

def jeffreys_der(x):
    return np.choose(x > 0, [np.inf, (x - 1)/x + np.log(x) ])

def jeffreys_conj(x):
    lambert = np.real(sp.special.lambertw(np.e**(1 - x)))
    return x - 2 + lambert + 1/lambert

def jeffreys_conj_der(x):
    return 1 / np.real(sp.special.lambertw(np.e**(1 - x)))

def chi_alpha(x, alpha):
    return np.choose(x >= 0, [np.inf, np.abs(x - 1) ** alpha])

def chi_alpha_Der(x, alpha):
    return np.choose(x >= 0, [np.inf, alpha * np.abs(x - 1)**(alpha - 1) * np.sgn(x - 1)])

def chi_alpha_conj(x, alpha):
    return np.choose(x >= - alpha, [-1, x + (alpha - 1) * (np.abs(x) / alpha)**(alpha/(alpha - 1))])

## divergences with non-finite conjugates
def reverse_kl(x, alpha):
    return np.choose(x > 0, [np.inf, x - 1 - np.log(x)])

def reverse_kl_der(x, alpha):
    return np.choose(x > 0, [np.inf, (x-1)/x])

def jensen_shannon(x, alpha):
    return np.choose(x > 0, [np.inf, np.log(x) - (x + 1) * np.log((x+1)/2) ])

def jensen_shannon_der(x, alpha):
    return np.choose(x > 0, [np.inf, 1/x - 1 - np.log((x+1)/2)])

def reverse_pearson(x, alpha):
    return np.choose(x > 0, [np.inf, 1/x - 1])

def reverse_pearson_der(x, alpha):
    return np.choose(x > 0, [np.inf, - 1/x**2])

def tv(x, alpha):
    return np.choose(x >= 0, [np.inf, np.abs(x - 1)])

def tv_der(x, alpha):
    return np.choose(x >= 0, [np.inf, np.sign(x - 1)])

def tv_conj(y, alpha):
    return np.choose(y <= 1, [np.inf, np.max(y, - 1)])

def tv_conj_der(y, alpha):
    return np.choose(np.abs(y) <= 1, [0, 1])



def DALE_flow(#prior, # initial configuration of particles
              #target, # target configuration of particles
              # alpha = 5, # Tsallis parameter
              # sigma = .501, # kernel parameter
              # N = 300, # number of samples
              # lambd = .01, # regularization parameter
              # step_size = .001, # step size for Euler scheme
              # max_time = 50, # maximal time horizon (i.e. simulate flow until T = max_time)
              # plot = True, # decide whether flow should be plotted
              # arrows = True, # decide whether to plot arrows corresponding to gradient vectors
              # gif = True, # decide whether to make a gif from simulation
              # timeline = True, # decide whether to plot timeline of DALE value along iteration
              # d = 2, # dimension of samples
              # kern = inv_multiquadric,
              # kern_der = inv_multiquadric_der,
              # mode = 'dual', # decide whether to solve the primal or dual problem
              # div_conj = tsallis_conj,
              # div_conj_der = tsallis_conj_der,
              # div = tsallis,
              # div_der = tsallis_der,
              # target_name = 'two_lines' # choose the target measure nu
              ):
    
    '''
    @return:    DALE_value:    list of length N, records objective value during simulation
                elapsed_time:  elapsed time during the simulation in seconds
                times:         list of length iterations, time it took solving the minimization, in seconds
                iteration:     list of length iterations, number of iterations it took solving the minimization, in seconds
    '''
    p_start = time.time()    
    #lambd = 1/np.sqrt(N) # (see bottom of p. 15 of KALE paper)
    #step_size = np.round(.1 * lambd, 5)
    alpha = 3
    sigma = .05
    N = 999
    lambd = .001
    step_size = .0001
    max_time = 10
    poster = True
    plot = True
    arrows = False
    gif = True
    timeline = True
    d = 2
    kern = inv_multiquadric
    kern_der = inv_multiquadric_der
    mode = 'dual'
    div_conj = tsallis_conj 
    div_conj_der = tsallis_conj_der
    div = tsallis
    div_der = tsallis_der
    target_name = 'circles'

    iterations = int(max_time / step_size) + 1, # max number of iterations


    if div != tsallis:
        alpha = ''
    
    kernel = kern.__name__
    divergence = div.__name__
    
    
    if plot or gif or timeline:
        folder_name = f"{divergence},alpha={alpha},lambd={lambd},tau={step_size},{kernel},{sigma},{N},{mode},{max_time},{target_name}"
        # Create the new folder in the current directory
        try:
            os.mkdir(folder_name)
            print(f"Folder '{folder_name}' created successfully.")
        except FileExistsError:
            print(f"Folder '{folder_name}' already exists.")
        except Exception as e:
            print(f"An error occurred: {e}.")
        
    # generate prior and target
    
    # prior = np.array([(k, 1) for k in np.linspace(-2, 2, N)])

    
    if target_name == 'circles': # multiple circles target
        #TODO: introduce parameter controlling number of rings. At the moment: 3 rings
        Ntilde, r, delta = int(N/3), 2, .25
        L = np.linspace(0, 2 * np.pi, Ntilde + 1)
        X = np.c_[r * np.sin(L), r * np.cos(L)][:-1]
        for i in [1, 2]: # for more circles, add more integers to this list
            X = np.r_[X, X[:Ntilde, :]-i*np.array([0, (2 + delta) * r])]
        target = X
        target[:, [0, 1]] = target[:, [1, 0]] # Exchange the columns
        
        # mean and variance of prior and target distributions
        m_p = np.array([-2.25, 0])
        v_p = 1/2000*np.array([[1, 0], [0, 1]])
        prior = np.random.multivariate_normal(m_p, v_p, N)

    
    if target_name == 'two_lines': # two lines target
        u = int(N/2)
    
        # layer 1
        vert = np.random.rand(u)
        hori = np.random.rand(u)
        l = np.linspace(-1, 1, u) + vert
        squared  = l**2 + hori
    
        # layer 2
        vert2 = np.random.rand(u)
        hori2 = np.random.rand(u)
        l2 = np.linspace(-1.5, 1.5, u) + vert2
        squared2  = 1/2*(l2-1)**2 + hori2 - 6
    
        l = np.append(l, l2)
        squared = np.append(squared, squared2)
        target = np.array([l, squared]).T
        # Exchange the columns
        target[:, [0, 1]] = target[:, [1, 0]]

        
        # mean and variance of prior and target distributions
        m_p = np.array([4, 0])
        v_p = 1/2000*np.array([[1, 0], [0, 1]])
        prior = np.random.multivariate_normal(m_p, v_p, N)
    
    if d == 2:
        prior_x, prior_y = prior.T
        target_x, target_y = target.T
    
    Y = prior.copy() # samples of prior distribution
    X = target # samples of target measure
    p_end = time.time()
    print(f"Generating prior and target took {p_end - p_start} seconds")

        
    # now start particle descent
    # Keeping track of different values along the iterations
    # DALE_value = np.zeros(iterations) # objective value during the algorithm
    # if mode == 'primal':
    #     primal_times = []
    #     primal_iterations = []
    # if mode == 'dual':
    #     dual_times = []
    #     dual_iterations = []
        
    iterations = int(iterations[0]) # reset iterations to be the int from the beginning
      
    
    start = time.time()
    for n in range(iterations):
        print(f'--------------------------- Iteration {n} ---------------------------')
        start_time = time.time()
        Z = np.append(X, Y, axis=0) # concatenate sample of target and positions of particles in step n
        M = len(Z)
        

        k_start = time.time()
        dist = squareform(pdist(Z, 'euclidean'))
        if kern == inv_multiquadric:
            K = (sigma + dist**2)**(-1/2)
        if kern == gauss:
            K = np.exp(-dist**(2) / (2 * sigma)) # print("Kernel matrix computed")
        k_end = time.time()
        print(f"Kernel matrix computed in {k_end - k_start} seconds")
        

        # this is minus the value of the DALE objective, if you multiply it by (1 + lambd)
        def prim_objective(b):
            p = K @ b        
            c1 = np.concatenate( (div_conj(p[:N], alpha = alpha), - p[N:]))
            c3 = b.T @ p
            return 1/N * np.sum(c1) + lambd/2 * c3
        
        # jacobian of the above ojective function
        def prim_jacobian(b):
            p = K @ b
            x = np.concatenate( (div_conj_der(p[:N], alpha = alpha), - np.ones(N)), axis=0)
            return 1/N * K @ x + lambd * p
        
        
        # dual objective is an N-dimensional function
        def dual_objective(q):
            convex_term = np.sum(div(q, alpha = alpha))
            tilde_q = np.concatenate((q, - np.ones(N)))
            quadratic_term = tilde_q.T @ K @ tilde_q
            return 1/N * convex_term + 1/(2 * lambd * N * N) * quadratic_term
    
        # jacobian of the above ojective function
        def dual_jacobian(q):
            convex_term = div_der(q, alpha = alpha)
            tilde_q = np.concatenate((q, - np.ones(N)))
            linear_term = K[:N, :] @ tilde_q
            return 1/N * convex_term + 1/(lambd * N * N) * linear_term
            
        # if mode == 'primal':
        if n == 0: # warm start
            # start = time.time()
            warm_start = np.concatenate((np.zeros(N), 1/(lambd*N) * np.ones(N)))
            # end = time.time()
            # if timeline:
            #     primal_times.append(end-start)
            #     primal_iterations.append(prob.nit)
        else:
            warm_start = beta
            
        prob = sp.optimize.minimize(prim_objective, warm_start, method='l-bfgs-b', jac = prim_jacobian, options={'gtol': 1e-9})
        beta = prob.x
            
                    
        h_star_grad = np.zeros((N, 2))
        for k in range(N):
            # TODO: In order to vectorize the following line, modify kern_der?
            temp = [beta[j] * kern_der(Y[k], Z[j], s = sigma) for j in range(M)]
            h_star_grad[k,:] = np.sum(temp, axis=0)
        
        # gradient update
        Y -= step_size * (1 + lambd) * h_star_grad
        #     DALE_value[n] = - (1 + lambd) * prob.fun
        
        
        # if mode == 'dual':
        sp_start = time.time()
        if n > 0: # warm start
            warm_start = q 
        else:
            warm_start = np.ones(N)
            
        prob = sp.optimize.minimize(dual_objective, warm_start, method='l-bfgs-b', jac = dual_jacobian, options={'gtol': 1e-15})
        sp_end = time.time()
        print(f"Scipy took {sp_end - sp_start} seconds")
        # if timeline:
        #     dual_times.append(sp_end-sp_start)
        #     dual_iterations.append(prob.nit)

        q = prob.x
        print(q.sum())
        print(beta.sum())
        
        plt.plot(beta, label='beta')
        q_new = 1/lambd * 1/N * np.concatenate((-q, np.ones(N)))
        plt.plot(q_new, label=r'$\frac{1}{\lambda N} [-q, 1]$')
        plt.legend()
        plt.title(f'Iteration {n}')
        plt.savefig(folder_name + f'qVsbeta-{n}.png', dpi=300)
        plt.show()


        #plt.plot(q), plt.show()
        
        ### gradient update     
        start2 = time.time()

        # h_star_grad = np.zeros((N, d))
        # for k in range(N): # TODO: vectorize the following line.
        #     temp = [kern_der(X[j], Y[k], sigma) - q[j] * kern_der(Y[j], Y[k], sigma) for j in range(N)]
        #     h_star_grad[k,:] = 1/(lambd * N) * np.sum(temp, axis=0)
        # # print(h_star_grad)
        # # plt.scatter(h_star_grad.T[0], h_star_grad.T[1]), plt.title('h_star_grad'), plt.show()
        # Y -= step_size * (1 + lambd) * h_star_grad
    
        end2 = time.time()
        print(f"Gradient update took {end2-start2} seconds")
        
        # DALE_value[n] = (1 + lambd) * prob.fun

        
        ### TODO: plot the initial configuration        

        # plot the particles ten times per unit time interval
        if plot:# and not n % int(1/(10*step_size)): # time1 in [0.0, 1.0, 2.0, 10.0, 50.0, 100.0]:
            time1 = round(n*step_size, 1)
            Y_1, Y_2 = Y.T
            plt.figure()
            plt.plot(target_x, target_y, '.', color='orange')#, label = 'target') 
            for i in range(len(Y)):
                point = Y[i]
                plt.plot(point[0], point[1], '.', c = 'b')
                if arrows:
                    vector = - h_star_grad[i]
                    # magnitude_v = np.linalg.norm(vector)
                    # Add an arrow from the point in the direction of the vector
                    plt.arrow(point[0], point[1], vector[0], vector[1], head_width=0.05, head_length=0.1, fc='k', ec='k', linewidth=.5)
    
    
            # plt.plot(Y_1, Y_2, 'x') #, label = 'Particles at \n' + fr'$t =${time}')
            if not poster:
                plt.plot([], [], ' ', label = fr't = {time1}') # Create empty plot with blank marker containing the extra label
                # plt.plot(prior.T[1], prior.T[0], 'x', label = 'prior')
                plt.legend(frameon = False)
           
                plt.title(f'{divergence}-DALE particle flow solved via the ' + mode + ' problem, \n' + fr'$\alpha$ = {alpha}, $\lambda$ = {lambd}, $\tau$ = {step_size}, $N$ = {N}, ' + kernel + fr' $s =$ {sigma}')
            if target_name == 'circles':
                plt.ylim([-3.5, 3.5])
                plt.xlim([-12, 2.5])
            plt.gca().set_aspect('equal')
            plt.axis('off')
            
            time_stamp = int(time1*10)
            plt.savefig(folder_name + f'/{divergence},alpha={alpha},lambd={lambd},tau={step_size},{kernel},{sigma},{N},{mode},{max_time},{target_name}-{time_stamp}.png', dpi=300)
            plt.show()
            plt.close()
            
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    with open(f'{divergence}_DALE_value,{alpha},{lambd},{step_size},{kernel},{sigma},{mode},{N},{max_time}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(DALE_value)
    
    if gif:
        output_name = f'{divergence},alpha={alpha},lambd={lambd},tau={step_size},{kernel},{sigma},{N},{mode},{max_time},{target_name}.gif'    
        create_gif(folder_name, output_name)
    
    if timeline:
        # plot DALE_value along the simulation
        plt.plot(np.array(range(iterations))*lambd/10, DALE_value)
        title1 = f'{divergence}-DALE objective value. Problem solved via the ' + mode + ' problem, \n'
        title2 = fr'$\alpha$ = {alpha}, $\lambda$ = {lambd}, $\tau$ = {step_size}, $N$ = {N},' + kernel + fr' kernel, $s =$ {sigma}'
        plt.title(title1 + title2)
        #plt.yscale('log')
        plt.xlabel(r'time $t$')
        plt.ylabel(fr'{divergence}-DALE$^{{({lambd})}}(\mu \mid \nu)$')
        plt.savefig(folder_name + f'/{divergence}_DALE_value_timeline,{alpha},{lambd},{step_size},{kernel},{sigma},{mode}.png', dpi=300)
        plt.show()
    
    if mode == 'primal':
        return DALE_value, elapsed_time #, primal_times, primal_iterations 
    if mode == 'dual':
        return DALE_value, elapsed_time #, dual_times, dual_iterations
    
    # L = np.array(range(n))
    # plt.plot(L, primal_times, label='primal')
    # plt.plot(L, dual_times, label='dual')
    # plt.legend()
    # plt.ylabel('time')
    # plt.xlabel('iterations')
    # plt.title(f'Computation time of L-BFGS in the {divergence}_DALE \n with parameters $\alpha =$ {alpha}, $\lambda = $ {lambd}, $\tau =$ {step_size},{kernel},{sigma}')
    # plt.savefig(f'Comparisons/Primal_Dual_Computation_Time_Comparison,{alpha},{lambd},{step_size},{kernel},{sigma},{mode}', dpi = 300)
    # plt.show()
    
    
    # plt.plot(L, primal_iterations, label='primal')
    # plt.plot(L, dual_iterations, label='dual')
    # plt.legend()
    # plt.title(f'Number of L-BFGS iterations in the {divergence}_DALE \n with parameters $\alpha =$ {alpha}, $\lambda = $ {lambd}, $\tau =$ {step_size},{kernel},{sigma}')
    # plt.ylabel('Number of iterations of sp.optimize.minimize')
    # plt.xlabel('iterations')
    # plt.savefig(f'Comparisons/Primal_Dual_Iteration_number_Comparison,{alpha},{lambd},{step_size},{kernel},{sigma},{mode}', dpi = 300)
    # plt.show()


# ### compare Tsallis-DALE flow for different alpha
# max_time = 100    
# N = 100
# divergence = 'tsallis'
# sigma = .05
# lambd = .01, # regularization parameter
# step_size = .001, # step size for Euler scheme
# kern = inv_multiquadric
# kernel = 'inverse multiquadric'
# mode = 'dual'

# #generate prior and target
# # mean and variance of prior and target distributions
# m_p, m_t = np.array([-2.25, 0]), np.array([-1, -1])
# v_p, v_t = 1/2000*np.array([[1, 0], [0, 1]]), 1/200*np.array([[1, 0], [0, 1]])
# prior = np.random.multivariate_normal(m_p, v_p, N)
# # prior = np.array([(k, 1) for k in np.linspace(-2, 2, N)])

# # # multiple circles target
# #TODO: introduce parameter controlling number of rings
# Ntilde, r, _delta = int(N/2), 2, .25
# X = np.c_[r * np.sin(np.linspace(0, 2 * np.pi, Ntilde + 1)), r * np.cos(np.linspace(0, 2 * np.pi, Ntilde + 1))][:-1]
# for i in [1]: # for more circles, add more integers to this list
#     X = np.r_[X, X[:Ntilde, :]-i*np.array([0, (2 + _delta) * r])]
# target = X
# # Exchange the columns
# target[:, [0, 1]] = target[:, [1, 0]]

# alphas = [1.5, 2, 5, 10]
# L = len(alphas)
# DALE_values = list(np.zeros(L))
# for (alpha, k) in zip(alphas, range(L)):
#     DALE_values[k] = DALE_flow(prior = prior, target = target, alpha = alpha, N=N, kern=inv_multiquadric, max_time=max_time, plot=False, timeline=False, gif=False)[0]

# lambd = lambd[0]
# step_size = step_size[0]
# for k in range(L):
#     plt.plot(np.linspace(0, max_time, len(DALE_values[0])), DALE_values[k], label = fr'$\alpha =$ {alphas[k]}')
# title1 = f'{divergence}-DALE objective value. Problem solved via the ' + mode + ' problem, \n'
# title2 = fr'$\lambda$ = {lambd}, $\tau$ = {step_size}, $N$ = {N}, ' + kernel + fr' kernel, $s$ = {sigma}'
# plt.title(title1 + title2)
# plt.legend(frameon = False)
# plt.yscale('log')
# plt.xlabel(r'time $t$')
# plt.ylabel(fr'{divergence}-DALE$^{{({lambd})}}(\mu \mid \nu)$')
# plt.savefig(f'{divergence}_DALE_value_timeline,{lambd},{step_size},{kernel},{sigma},{mode},{max_time}.png', dpi=300)
# plt.show()