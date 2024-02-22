import os
from warnings import warn
import torch
import ot
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from kernels import *
from adds import *
from entropies import *
from data_generation import *

torch.set_default_dtype(torch.float64) # set higher precision
use_cuda = torch.cuda.is_available() # shorthand
my_device = 'cuda' if use_cuda else 'cpu' 

def MMD_reg_f_div_flow(
        alpha = 5,
        sigma = .05,
        N = 1002,
        lambd = .01,
        step_size = .001,
        max_time = 1,
        plot = True, # plot particles along the evolution
        arrows = False, # plots arrows at particles to show their gradients
        timeline = True, # plots timeline of functional value along the iterative scheme
        d = 2, # dimension of the ambient space in which the particles live
        kern = IMQ,
        kern_der = IMQ_der,
        mode = 'primal',
        div_conj = tsallis_conj,
        div_conj_der = tsallis_conj_der,
        div = tsallis,
        div_der = tsallis_der,
        target_name = 'circles',
        verbose = False,
        compute_W2 = False,
        save_opts = False,
        compute_KALE = False,
        st = 42
        ):
    
    '''
    @return:    func_value:    torch tensor of length iterations, records objective value along the flow
                MMD:           torch tensor of length iterations, records MMD between particles along the flow
                W2:            torch tensor of length iteratiobs, records W2 metric between particles along the flow
                KALE_values:   torch tensor of length iterations, records regularized KL divergence between particles and target along the flow
    '''
    

    iterations = int(max_time / step_size) + 1, # max number of iterations


    if div != tsallis and div != chi:
        alpha = ''
    
    kernel = kern.__name__
    divergence = div.__name__
    # {divergence},lambda={lambd},tau={step_size},{kernel},{sigma},{N},{max_time},{target_name}/
    folder_name = f"{divergence},alpha={alpha},lambd={lambd},tau={step_size},{kernel},{sigma},{N},{mode},{max_time},{target_name},state={st}"
    make_folder(folder_name)
        
    # generate prior and target   
    if target_name == 'cross':
        samples = neals_funnel(int(N/4))
        samples1 = rotate_points(samples, 90)
        samples2 = rotate_points(samples, 180)
        samples3 = rotate_points(samples, 270)

        new_samples = np.append(samples, samples1, axis=0)
        new_samples = np.append(new_samples, samples2, axis=0)
        target = torch.from_numpy(np.append(new_samples, samples3, axis=0))
        # TODO: shorten the previous lines using torch.stack or so
        
        # mean and variance of prior distribution
        m_p = torch.zeros(d)
        v_p = 1/2000*torch.eye(d)
        
        # Draw samples from the prior distribution
        multivariate_normal = torch.distributions.MultivariateNormal(m_p, v_p) # Create a MultivariateNormal distribution
        prior = multivariate_normal.sample((N,)) # Generate samples
    
     
    if target_name == 'circles': # multiple circles target
        target, prior = generate_data(int(N/3), st=st)
        

    if target_name == 'bananas': # bananas target
        u = int(N/2)
        
        torch.manual_seed(st) # fix randomness
        
        # layer 1
        vert1 = torch.rand(u)
        hori1 = torch.rand(u)
        l1 = torch.linspace(-1, 1, u) + vert1
        squared1 = l1**2 + hori1
    
        # layer 2
        vert2 = torch.rand(u)
        hori2 = torch.rand(u)
        l2 = torch.linspace(-1.5, 1.5, u) + vert2
        squared2  = 1/2*(l2-1)**2 + hori2 - 4
        
        l = torch.cat((l1, l2))
        squared = torch.cat((squared1, squared2))
        target = torch.stack((l, squared)).transpose(0, 1)
        
        # mean and variance of prior distribution
        m_p = torch.tensor([0, 4.0])
        v_p = 1 / 2000 * torch.eye(d)


        # Draw samples from the prior distribution
        multivariate_normal = torch.distributions.MultivariateNormal(m_p, v_p) # Create a MultivariateNormal distribution
        prior = multivariate_normal.sample((N,)) # Generate samples
     
        
    Y = prior.clone().to(my_device) # samples of prior distribution
    X = target.to(my_device) # samples of target measure 
    torch.save(X, folder_name + f'/target.pt')
      

    #### now start particle descent
    iterations = int(iterations[0]) # reset iterations to be the int from the beginning
    func_values = [] # objective value during the algorithm
    KALE_values = torch.zeros(iterations)
    dual_values = []
    pseudo_dual_values = []
    MMD = torch.zeros(iterations) # mmd(X, Y) during the algorithm
    W2 = torch.zeros(iterations)
    duality_gaps = []
    pseudo_duality_gaps = []
    relative_duality_gaps = []
    relative_pseudo_duality_gaps = []   
    
    kxx = kern(X[:, None, :], X[None, :, :], sigma)
    
    if compute_W2: 
        a, b = torch.ones(N) / N, torch.ones(N) / N

    for n in range(iterations):
        # plot the particles ten times per unit time interval
        if plot and not n % 1000 or n in 100*np.arange(1, 10):
            Y_cpu = Y.cpu()
            plt.figure() 
            plt.plot(target[:, 1], target[:, 0], '.', color='orange', markersize = 2) # plot target
            plt.plot(Y_cpu[:, 1], Y_cpu[:, 0], '.', color='blue', markersize = 2) # plot particles
            if arrows and n > 0:
                minus_grad_cpu = - h_star_grad.cpu()
                plt.quiver(Y_cpu[:, 1], Y_cpu[:, 0], minus_grad_cpu[:, 1], minus_grad_cpu[:, 0], angles='xy', scale_units='xy', scale=1)
                '''
                for i in range(Y.shape[0]):
                    point = Y_cpu[i, :]
                    vector = - h_star_grad.cpu()[i]
                    plt.arrow(point[1], point[0], vector[1], vector[0], head_width=0.05, head_length=0.1, fc='k', ec='k', linewidth=.5)                        
                '''
            
            if target_name == 'circles':
               plt.ylim([-.5, .5])
               plt.xlim([-2.0, .5])

            plt.gca().set_aspect('equal')
            plt.axis('off')
            
            img_name = f'/Reg_{divergence}{alpha}flow,lambd={lambd},tau={step_size},{kernel},{sigma},{N},{max_time},{target_name}-{n}.png'
            plt.savefig(folder_name + img_name, dpi=300, bbox_inches='tight')
            plt.close()
    
    
        ### construct kernel matrix
        kxy = kern(X[:, None, :], Y[None, :, :], sigma)
        kyy = kern(Y[:, None, :], Y[None, :, :], sigma)
        upper_row = torch.cat((kxx, kxy), dim=1)
        lower_row = torch.cat((kxy.t(), kyy), dim=1)
        K = torch.cat((upper_row, lower_row), dim=0)
        K = K.cpu()
        K = K.numpy()
        
        ## calculate MMD(X, Y), W1 and W2 metric between particles and target
        MMD[n] = 1/(2 * N**2) * (kxx.sum() + kyy.sum() - 2 * kxy.sum())
        if compute_W2:
            M2 = ot.dist(X, Y, metric='sqeuclidean')
            W2[n] = ot.emd2(a, b, M2)


        # primal objective is an N-dimensional function
        def primal_objective(q):
            convex_term = np.sum(div(q, alpha))
            tilde_q = np.concatenate((q, - np.ones(N)))
            quadratic_term = tilde_q.T @ K @ tilde_q
            return 1/N * convex_term + 1/(2 * lambd * N * N) * quadratic_term

        # jacobian of the above ojective function
        def primal_jacobian(q):
            convex_term = div_der(q, alpha)
            tilde_q = np.concatenate((q, - np.ones(N)))
            linear_term = upper_row.cpu().numpy() @ tilde_q
            return 1/N * convex_term + 1/(lambd * N * N) * linear_term

        def primal_KALE_objective(q):
            convex_term = np.sum(div(q, 1))
            tilde_q = np.concatenate((q, - np.ones(N)))
            quadratic_term = tilde_q.T @ K @ tilde_q
            return 1/N * convex_term + 1/(2 * lambd * N * N) * quadratic_term

        # jacobian of the above ojective function
        def primal_KALE_jacobian(q):
            convex_term = div_der(q, 1)
            tilde_q = np.concatenate((q, - np.ones(N)))
            linear_term = upper_row.cpu().numpy() @ tilde_q
            return 1/N * convex_term + 1/(lambd * N * N) * linear_term

            
        # this is minus the value of the objective, if you multiply it by (1 + lambd)
        def dual_objective(b):
            p = K @ b        
            c1 = np.concatenate( (div_conj(p[:N], alpha), - p[N:]))
            c3 = b.T @ p
            return 1/N * np.sum(c1) + lambd/2 * c3
        
        # jacobian of the above ojective function
        def dual_jacobian(b):
            p = K @ b
            x = np.concatenate( (div_conj_der(p[:N], alpha), - np.ones(N)), axis=0)
            return 1/N * K @ x + lambd * p

        if n > 0: # warm start
            warm_start_q = q_np # take solution from last iteration
            if mode == 'dual' and not alpha == '':
                warm_start_b = 1/(lambd*N) * np.concatenate((- q_np, np.ones(N)))
            
        else:
            warm_start_q = 1/1000*np.ones(N)
            if mode == 'dual' and not alpha == '':
                warm_start_b = 1/(lambd*N) * np.concatenate((- warm_start_q, np.ones(N)))
        
        opt_kwargs = dict(
            m=100,
            factr=100,
            pgtol=1e-7,
            iprint=0,
            maxiter=120,
            disp=0,
        ) 
        q_np, prim_value, _ = sp.optimize.fmin_l_bfgs_b(
            primal_objective,
            warm_start_q,
            fprime=primal_jacobian,
            bounds=[(0, None) for _ in range(N)],
            **opt_kwargs,
        )
        if compute_KALE:
            _, prim_value_KALE, _ = sp.optimize.fmin_l_bfgs_b(
                primal_KALE_objective,
                warm_start_q,
                fprime=primal_KALE_jacobian,
                bounds=[(0, None) for _ in range(N)],
                **opt_kwargs,
            )
            KALE_values[n] = prim_value_KALE

        func_values.append(prim_value)
        if mode == 'dual' and alpha == '':
            b_np, minus_dual_value, _ = sp.optimize.fmin_l_bfgs_b(dual_objective, warm_start_b, fprime = dual_jacobian, **opt_kwargs) 
            dual_values.append(-minus_dual_value)
            if plot and save_opts and not n % 10000:
              torch.save(torch.from_numpy(b_np), f'{folder_name}/b_at_{n}.pt')

        
        pseudo_dual_value = - dual_objective(1/(lambd * N) * np.concatenate((-q_np, np.ones(N))))
        pseudo_dual_values.append(pseudo_dual_value)
        pseudo_duality_gap = np.abs(prim_value - pseudo_dual_value)
        pseudo_duality_gaps.append(pseudo_duality_gap)
        relative_pseudo_duality_gap = pseudo_duality_gap / np.min((np.abs(prim_value), np.abs(pseudo_dual_value)))
        relative_pseudo_duality_gaps.append(relative_pseudo_duality_gap)
        pseudo_gap_tol = 1e-2
        relative_pseudo_gap_tol = 1e-2
        if pseudo_duality_gap > pseudo_gap_tol and verbose:
              warn(f'Iteration {n}: pseudo-duality gap {pseudo_duality_gap} is larger than tolerance {pseudo_gap_tol}.')
        if relative_pseudo_duality_gap > relative_pseudo_gap_tol and verbose:
              warn(f'Iteration {n}: relative pseudo-duality gap {relative_pseudo_duality_gap} is larger than tolerance {relative_pseudo_gap_tol}.')
              
        if mode == 'dual' and not alpha == '':
          dual_value = - minus_dual_value
          duality_gap = np.abs(prim_value - dual_value)
          duality_gaps.append(duality_gap)
          relative_duality_gap = duality_gap / np.min((np.abs(prim_value), np.abs(dual_value)))
          relative_duality_gaps.append(relative_duality_gap)
          gap_tol = 1e-2
          relative_gap_tol = 1e-2  
          if relative_duality_gap > relative_gap_tol and verbose:
              warn(f'Iteration {n}: relative duality gap {relative_duality_gap} is larger than tolerance {relative_gap_tol}.')      

        q_torch = torch.tensor(q_np, dtype=torch.float64, device=my_device)

        # save solution vector in every 100-th iteration (to conserve memory)
        if plot and save_opts and not n % 1000:
              torch.save(q_torch, f'{folder_name}/q_at_{n}.pt')
              
        # gradient update
        temp = kern_der(Y, Y, sigma) - q_torch.view(N, 1, 1) * kern_der(Y, X, sigma)
        h_star_grad = 1 / (lambd * N) * torch.sum(temp, dim=0)            
        if plot and save_opts and not n % 1000:
          torch.save(h_star_grad, f'{folder_name}/h_star_grad_at_{n}.pt')
        Y -= step_size * (1 + lambd) * h_star_grad
        # save position of particles in every 100-th iteration (to conserve memory)
        if not n % 1000 or n in 100*np.arange(1, 10):
            torch.save(Y, f'{folder_name}/Y_at_{n}.pt')
    
    
    suffix = f',{lambd},{step_size},{N},{kernel},{sigma},{max_time},{target_name}'
    torch.save(func_values, folder_name + f'/Reg_{divergence}-{alpha}_Div_value_timeline{suffix}.pt')
    torch.save(MMD, folder_name + f'/Reg_{divergence}-{alpha}_Div_MMD_timeline{suffix}.pt')
    if compute_W2:
        torch.save(W2, folder_name + f'/Reg_{divergence}-{alpha}_DivW2_timeline{suffix}.pt')
    if mode == 'dual':
        torch.save(duality_gaps, folder_name + f'/Reg_{divergence}-{alpha}_Divergence_duality_gaps_timeline{suffix}.pt')
        torch.save(relative_duality_gaps, folder_name + f'/Reg_{divergence}-{alpha}_Divergence_rel_duality_gaps_timeline{suffix}.pt')
    torch.save(pseudo_duality_gaps, folder_name + f'/Reg_{divergence}-{alpha}_Divergence_pseudo_duality_gaps_timeline{suffix}.pt')
    torch.save(relative_pseudo_duality_gaps, folder_name + f'/Reg_{divergence}-{alpha}_Divergence_rel_pseudo_duality_gaps_timeline{suffix}.pt')       
    if timeline:
        # plot MMD, objective value, and W2 along the flow
        fig, ax = plt.subplots()
        plt.plot(MMD.cpu().numpy())
        plt.xlabel('iterations')
        plt.ylabel(r'$d_{K}(\mu, \nu)$')
        plt.yscale('log')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8)))
        plt.savefig(folder_name + f'/{divergence}_MMD_timeline,{alpha},{lambd},{step_size},{kernel},{sigma}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
        # Plot functional values
        fig, ax = plt.subplots()
        if not alpha == '':
            plt.plot(dual_values, label='dual objective')
        else:
            plt.plot(pseudo_dual_values, label='pseudo dual values')
        plt.plot(func_values, '--', label='primal objective')
        plt.yscale('log')
        plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8)))
        plt.xlabel('iterations')
        plt.ylabel(r'$D_{f_{\alpha}}^{{' + str(lambd) + r'}}(\mu \mid \nu)$')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend(frameon=False)
        plt.savefig(folder_name + f'/{divergence}_objective_timeline,{alpha},{lambd},{step_size},{kernel},{sigma}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        if compute_W2:     
          fig, ax = plt.subplots()
          plt.plot(W2.cpu().numpy())
          plt.yscale('log')
          plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8)))
          plt.xlabel('iterations')
          plt.ylabel(r'$W_2(\mu, \nu)$')
          ax.spines['top'].set_visible(False)
          ax.spines['right'].set_visible(False)
          plt.savefig(folder_name + f'/{divergence}_W2_timeline,{alpha},{lambd},{step_size},{kernel},{sigma}.png', dpi=300, bbox_inches='tight')
          plt.close()
        
        # plot pseudo and relative duality gaps
        fig, ax = plt.subplots()
        if not alpha == '': # and alpha > 1:
          plt.plot(duality_gaps, label='duality gap')
          plt.plot(relative_duality_gaps, '-.', label='relative duality gap')
        plt.plot(pseudo_duality_gaps, ':', label='pseudo duality gap')
        plt.plot(relative_pseudo_duality_gaps, label='relative pseudo-duality gap')
        plt.axhline(y=1e-2, linestyle='--', color='gray', label='tolerance')
        plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8)))
        plt.yscale('log')
        plt.xlabel('iterations')
        plt.legend(frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.savefig(folder_name + f'/{divergence}_duality_gaps_timeline,{alpha},{lambd},{step_size},{kernel},{sigma}.png', dpi=300, bbox_inches='tight')
        plt.close()   
        
        func_values = torch.tensor(np.array(func_values))
        KALE_values = torch.tensor(np.array(KALE_values))

    return func_values, MMD, W2, KALE_values


def this_main(
    sigma = .5,
    step_size = 1e-1,
    max_time = 1,
    lambd = 1e-0,
    N = 300*3,
    kern = IMQ,
    kern_der = IMQ_der,
    target_name = 'bananas',
    alphas = [3], #, 3/2, 2, 5/2, 3, 4, 5, 15/2, 10],
    div = tsallis,
    div_der = tsallis_der,
    div_conj = tsallis_conj,
    div_conj_der = tsallis_conj_der,
    compute_W2 = False,
    compute_KALE = False
    ):
    if div != tsallis and div != chi:
        alpha = ''
    kernel = kern.__name__
    diverg = div.__name__
    iterations = int(max_time / step_size)
    # states = [0, 1, 2, 3 ,4]
    L = len(alphas)
    func_values = torch.zeros(L, iterations + 1)
    MMD_values = torch.zeros(L, iterations + 1)
    W2_values = torch.zeros(L, iterations + 1)
    KALE_values = torch.zeros(L, iterations + 1)
     
    folder = f'{diverg},lambda={lambd},tau={step_size},{kernel},{sigma},{N},{max_time},{target_name}'
    make_folder(folder)
    
    for k in range(L):
      func_values[k, :], MMD_values[k, :], W2_values[k, :], KALE_values[k, :] = MMD_reg_f_div_flow(
            div = div,
            div_der = div_der,
            div_conj = div_conj,
            div_conj_der = div_conj_der,
            max_time = max_time,
            alpha = alphas[k],
            N = N,
            lambd = lambd,
            sigma = sigma,
            step_size = step_size,
            kern = kern,
            kern_der = kern_der,
            verbose = False,
            target_name = target_name,
            plot=True, timeline=True, arrows=False, compute_W2 = compute_W2, compute_KALE = compute_KALE) #, st = k)
           
    torch.save(func_values, f'{folder}/Reg_{diverg}_Div_value_timeline,{lambd},{step_size},{N},{kernel},{sigma},{max_time},{target_name}.pt')
    torch.save(MMD_values, f'{folder}/Reg_{diverg}_Div_MMD_timeline,{lambd},{step_size},{N},{kernel},{sigma},{max_time},{target_name}.pt')
    torch.save(W2_values, f'{folder}/Reg_{diverg}_Div_W2_timeline,{lambd},{step_size},{N},{kernel},{sigma},{max_time},{target_name}.pt')
    
    
    fig, ax = plt.subplots()
    for k in range(L):
        plt.plot(func_values[k, :], label = f'{alphas[k]}')
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel(r'$D_{f_{3}}^{{' + str(lambd) + r'}}(\mu \mid \nu)$')
    plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(2,10)*1/10))
    plt.legend(frameon=False, facecolor='white', framealpha=1, title=r'$\alpha$')
    plt.grid(which='both', color='gray', linestyle='--', alpha=.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(f'{folder}/Reg_{diverg}_Div_timeline,{step_size},{N},{kernel},{sigma},{max_time},{target_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # plot MMD
    fig, ax = plt.subplots()
    for k in range(L):
        plt.plot(MMD_values[k, :], label = f'{alphas[k]}')
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel(r'$d_K(\mu, \nu)^2$')
    plt.legend(frameon=False, facecolor='white', framealpha=1, title=r'$\alpha$')
    plt.grid(which='both', color='gray', linestyle='--', alpha=.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(f'{folder}/Reg_{diverg}_Div_MMD_timeline,{step_size},{N},{kernel},{sigma},{max_time},{target_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    if compute_KALE:
      fig, ax = plt.subplots()
      for k in range(L):
          plt.plot(KALE_values[k, :], label = f'{alphas[k]}')
      plt.yscale('log')
      plt.xlabel('iterations')
      plt.ylabel(r'KALE$(\mu, \nu)$')
      plt.legend(frameon=False, facecolor='white', framealpha=1, title=r'$\alpha$')
      plt.grid(which='both', color='gray', linestyle='--', alpha=.25)
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      plt.savefig(f'{folder}/Reg_{diverg}_Div_KALE_timeline,{step_size},{N},{kernel},{sigma},{max_time},{target_name}.png', dpi=300, bbox_inches='tight')
      plt.close()
      
    if compute_W2:
        fig, ax = plt.subplots()
        for k in range(L):
            plt.plot(W2_values[k, :], label = f'{alphas[k]}')
        plt.yscale('log')
        plt.xlabel('iterations')
        plt.ylabel(r'$W_{2}(\mu, \nu)$')
        plt.legend(frameon=False, facecolor='white', framealpha=1, title=r'$\alpha$')
        plt.grid(which='both', color='gray', linestyle='--', alpha=.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.savefig(f'{folder}/Reg_{diverg}_Div_W2_timeline,{step_size},{N},{kernel},{sigma},{max_time},{target_name}.png', dpi=300, bbox_inches='tight')
        plt.close()