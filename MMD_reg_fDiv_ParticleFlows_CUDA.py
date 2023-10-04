# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 13:51:56 2023

@author: vglom
"""
import os
from PIL import Image

import torch
import scipy
import numpy as np

import math
import time
import matplotlib.pyplot as plt
import csv

#import numpy as np

def generate_points(N, n, r, delta):
    if N % n != 0:
        raise ValueError("N must be divisible by n for even distribution.")

    points_per_ring = N // n
    points = []

    for i in range(n):
        theta_values = torch.linspace(0, 2 * math.pi, points_per_ring + 1)[:-1]
        radius_values = torch.full((points_per_ring,), r)

        # Calculate x-coordinates with spacing delta between rings
        x = radius_values * torch.cos(theta_values) - i * (2+delta)*r
        y = radius_values * torch.sin(theta_values)

        # Append the points to the result list
        points.append(torch.stack((x, y), dim=1))

    # Concatenate the points from all rings
    result = torch.cat(points, dim=0)

    return result


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


torch.manual_seed(2) # fix randomness


# Gaussian kernel with width s
def gauss(x, y, s):
    d = x - y
    return np.exp(- 1/(2 * s) * np.dot(d.T, d)) 

# derivative of Gaussian kernel    
def gauss_der(X, Y, s):
    return - 1 / s * (X - Y) * gauss(X, Y, s) 


def inv_multiquadric(x, y, s, b = 1/2):
    return ( ((x - y) ** 2).sum(axis=-1) + s )**(-b)

def inv_multiquadric_der(x, y, s = 0.05, b = 1/2):
    diff = x[:,None, :] - y[None,:, :]
    
    pref = - 2*b * (torch.norm(diff, dim=2, keepdim=True)**2 + s)**(-b-1)
    return pref * diff

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



def MMD_reg_f_div_flow(
        alpha = 5,
        sigma = .05,
        N = 300,
        lambd = .01,
        step_size = .001,
        max_time = 10,
        poster = True, # remove titles from plots
        plot = True, # plot particles along the evolution
        arrows = False, # plots arrows at particles to show their gradients
        gif = True, # produces gif showing the evolution of the particles
        timeline = True, # plots timeline of functional value along the iterative scheme
        d = 2, # dimension of the ambient space in which the particles live
        kern = inv_multiquadric,
        kern_der = inv_multiquadric_der,
        mode = 'dual',
        # div_conj = tsallis_conj,
        # div_conj_der = tsallis_conj_der,
        div = tsallis,
        div_der = tsallis_der,
        target_name = 'circles',
        verbose = False
        ):
    
    '''
    @return:    DALE_value:    list of length N, records objective value during simulation
                elapsed_time:  elapsed time during the simulation in seconds
                times:         list of length iterations, time it took solving the minimization, in seconds
                iteration:     list of length iterations, number of iterations it took solving the minimization, in seconds
    '''
    

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
    
    p_start = time.time()    
    # generate prior and target    
    if target_name == 'circles': # multiple circles target
        target = generate_points(N, 3, 2, .25)
    
        # mean and variance of prior distribution
        m_p = torch.tensor([-2.25, 0])
        v_p = 1 / 2000 * torch.tensor([[1, 0], [0, 1]])
        
        # Draw samples from the prior distribution
        multivariate_normal = torch.distributions.MultivariateNormal(m_p, v_p) # Create a MultivariateNormal distribution
        prior = multivariate_normal.sample((N,)) # Generate samples

    
    if d == 2:
        # prior_x, prior_y = prior.T[0], prior.T[1]
        target_x, target_y = target.T[0], target.T[1] 
        
    Y = prior.clone().to("cuda") # samples of prior distribution
    X = target.to("cuda") # samples of target measure #print("Prior and target constructed")
    
    p_end = time.time()
    print(f"Generating prior and target took {p_end - p_start} seconds")
        
    # now start particle descent
    # Keeping track of different values along the iterations
    DALE_value = torch.zeros(iterations) # objective value during the algorithm
    if mode == 'dual':
        dual_times = []
        dual_iterations = []
        
    iterations = int(iterations[0]) # reset iterations to be the int from the beginning
        
    start_time = time.time()
    
    # dist_xx = torch.cdist(X, X)

    for n in range(iterations):
        if verbose: print(f"Iteration -------- {n} ---------")
        k_start = time.time()
        Z = torch.cat((X, Y), axis=0) # concatenate sample of target and positions of particles in step n
        # M = len(Z)
        
        # dist_xy = torch.cdist(X, Y)
        # dist_yy = torch.cdist(Y, Y)
        # distances = torch.stack([dist_xx, dist_xy, dist_xy.t(), dist_yy])
        
        # construct kernel matrix
        distances = torch.cdist(Z, Z)
        
        if kern == inv_multiquadric:
            K = (distances.pow(2) + sigma).pow(-1/2).cpu().numpy()
        
        if kern == gauss:
            K = torch.exp(-distances.pow(2) / (2 * sigma**2)).cpu().numpy()
        k_end = time.time()
        # plt.imshow(K), plt.show()
        if verbose: print(f"Kernel matrix computed in {k_end-k_start} seconds")


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
            
        
        if mode == 'dual':
            sp_start = time.time()

            if n > 0: # warm start
                warm_start = q_np # take solution from last iteration
            else:
                warm_start = np.ones(N)
                
            result = scipy.optimize.minimize(dual_objective, warm_start, method='L-BFGS-B', jac=dual_jacobian)
            sp_end = time.time()
            if verbose: print(f"Scipy took {sp_end - sp_start} seconds")
            start_ne = time.time()
            q_np = result.x # solution vector of optimization problem
            # plt.plot(q_np), plt.show()
            q_torch = torch.tensor(q_np, dtype=torch.float64, device="cuda") # torch version of solution vector
            end_ne = time.time()
            if verbose: print(f"Rest took {end_ne - start_ne} seconds")
            # if timeline:
            #     dual_times.append(sp_end - sp_start)
            #     dual_iterations.append(dual_objective(q_np))

            # gradient update
            g_start = time.time()
            temp = kern_der(X, Y, sigma) - q_torch.view(N, 1, 1) * kern_der(Y, Y, sigma)
            h_star_grad = 1 / (lambd * N) * torch.sum(temp, dim=0)
            # plt.scatter(h_star_grad.T[0], h_star_grad.T[1]), plt.show()
            g_end = time.time()
            if verbose: print(f"Gradient update took {g_end - g_start} seconds")
            Y -= step_size * (1 + lambd) * h_star_grad
                                
        ### TODO: plot the initial configuration
        
        time1 = round(n*step_size, 1)

        # plot the particles ten times per unit time interval
        if plot and not n % int(1/(10*step_size)): # or:  time1 in [0.0, 1.0, 2.0, 10.0, 50.0, 100.0]:
            plot_start = time.time()
            Y_1, Y_2 = Y.cpu().T
            plt.figure()
            plt.plot(target_x, target_y, '.', color='orange', markersize = 2) #, label = 'target') 
            for i in range(len(Y)):
                point = Y[i]
                if arrows:
                    vector = - h_star_grad[i]
                    # magnitude_v = torch.norm(vector)
                    # Add an arrow from the point in the direction of the vector
                    plt.arrow(point[0], point[1], vector[0], vector[1], head_width=0.05, head_length=0.1, fc='k', ec='k', linewidth=.5)
    
    
            plt.plot(Y_1, Y_2, '.', markersize = 2) #, label = 'Particles at \n' + fr'$t =${time}')
            if not poster:
                plt.plot([], [], ' ', label = fr't = {time1}') # Create empty plot with blank marker containing the extra label
                # plt.plot(prior.T[1], prior.T[0], 'x', label = 'prior')
                plt.legend(frameon = False)
           
                plt.title(f'{divergence}-DALE particle flow solved via the ' + mode + ' problem, \n' + fr'$\alpha$ = {alpha}, $\lambda$ = {lambd}, $\tau$ = {step_size}, $N$ = {N}, ' + kernel + fr' $s =$ {sigma}')
            plt.gca().set_aspect('equal')
            plt.axis('off')
            
            time_stamp = int(time1*10)
            img_name = f'/{divergence}{alpha}_DALE_flow,lambd={lambd},tau={step_size},{kernel},{sigma},{N},{mode},{max_time},{target_name}-{time_stamp}.png'
            plt.savefig(folder_name + img_name, dpi=300, bbox_inches='tight')
            #plt.legend()
            plt.show()
            plt.close()
            plot_end = time.time()
            if verbose: print(f"Plotting took {plot_end - plot_start} seconds")
    end_time = time.time()
    elapsed_time = end_time - start_time
     
    if gif:
        output_name = img_name = f'/{divergence}{alpha}_DALE_flow,lambd={lambd},tau={step_size},{kernel},{sigma},{N},{mode},{max_time},{target_name}-{time_stamp}.gif'    
        create_gif(folder_name, output_name)
    
    if timeline:
        # plot DALE_value along the simulation
        plt.plot(torch.arange(iterations)*lambd/10, DALE_value)
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