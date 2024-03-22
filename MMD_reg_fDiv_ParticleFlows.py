import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from warnings import warn
import torch
import ot
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from kernels import *
from adds import *
from entropies import *
from data_generation import *

torch.set_default_dtype(torch.float64)  # set higher precision
use_cuda = torch.cuda.is_available()  # shorthand
my_device = 'cuda' if use_cuda else 'cpu'


def MMD_reg_f_div_flow(
        alpha=4,  # divergence parameter
        s=.1,  # kernel parameter
        N=900,  # number of target particles
        M=900,  # number of prior particles
        lambd=.01,  # regularization
        step_size=.001,  # step size for Euler forward discretization
        max_time=50,  # maximal time horizon for simulation
        plot=True,  # plot particles along the evolution
        arrows=False,  # plots arrows at particles to show their gradients
        timeline=True,  # plots timeline of functional value along the flow
        kern=imq,  # kernel
        dual=False,  # if dual==True, then the dual problem is solved as well
        div=tsallis,  # entropy function
        target_name='circles',  # name of the target measure nu
        verbose=False,  # decide whether to print warnings
        compute_W2=False,  # compute W2 dist of particles to target along flow
        save_opts=False,  # save minimizers and gradients along the flow
        compute_KALE=False,  # compute MMD-reg. KL-div. from particle to target
        st = 42,  # random state for reproducibility
        annealing=False,
        annealing_factor=0
        ):
    '''
    @return:    func_value:    torch tensor of length iterations, records objective value along the flow
                MMD:           torch tensor of length iterations, records 1/2 MMD^2 between particles along the flow
                W2:            torch tensor of length iterations, records W2 metric between particles along the flow
                KALE_values:   torch tensor of length iterations, records regularized KL divergence between particles and target along the flow
    '''

    iterations = int(max_time / step_size) + 1  # max number of iterations
    if not div in [tsallis, chi, lindsay, perimeter]:
        alpha = None

    kern_der = globals().get(kern.__name__ + '_der')
    kernel = kern.__name__
    B = emb_const(kern, s)  # embedding constant H_K \hookrightarrow C_0
    divergence = div.__name__
    div_reces = rec_const(divergence, alpha)  # recession constant of entropy function
    div_conj = globals().get(div.__name__ + '_conj')  # convex conjugate of the entropy function
    div_conj_der = globals().get(div.__name__ + '_conj_der')  # derivative of div_conj
    div_der = globals().get(div.__name__ + '_der')  # derivative of div

    folder_name = f"{divergence},alpha={alpha},lambd={lambd},tau={step_size},{kernel},{s},{N},{max_time},{target_name},state={st}_{annealing}={annealing_factor}"
    make_folder(folder_name)

    if verbose and B:
        print(f'Kernel is {kernel}, embedding constant is {round(B,2)}, recession constant is {div_reces}')

    target, prior = generate_prior_target(N, st, target_name) 
    torch.save(target, folder_name + f'/target.pt')

    Y = prior.clone().to(my_device)  # samples of prior distribution
    X = target.to(my_device)  # samples of target measure
    M = Y.shape[0]
    d = len(X[0])  # dimension of the ambient space in which the particles live
    func_values = []  # objective value during the algorithm
    KALE_values = torch.zeros(iterations)
    dual_values = []
    lambdas = []  # regularization parametre during the algorithm (relevant for annealing) 
    pseudo_dual_values = []
    MMD = torch.zeros(iterations)  # 1/2 mmd(X, Y)^2 during the algorithm
    W2 = torch.zeros(iterations)
    duality_gaps = []
    pseudo_duality_gaps = []
    relative_duality_gaps = []
    relative_pseudo_duality_gaps = []
    lower_bds_lambd = []

    kxx = kern(X[:, None, :], X[None, :, :], s)

    if compute_W2:
        a, b = torch.ones(N) / N, torch.ones(N) / N

    for n in range(iterations):
        # plot the particles
        if plot and not n % 1000 or n in 1e2*np.arange(1, 10):
            if annealing:
                img_name = f'/Reg_{divergence}{alpha}flow,annealing,tau={step_size},{kernel},{s},{N},{max_time},{target_name}-{n}.png'
            else:
                img_name = f'/Reg_{divergence}{alpha}flow,lambd={lambd},tau={step_size},{kernel},{s},{N},{max_time},{target_name}-{n}.png'

            Y_cpu = Y.cpu()
            if d == 2:
                plt.figure()
                plt.plot(target[:, 1], target[:, 0], '.', c='orange', ms=2)
                plt.plot(Y_cpu[:, 1], Y_cpu[:, 0], 'b.', ms=2)
                if arrows and n > 0:
                    minus_grad = - h_star_grad.cpu()
                    plt.quiver(Y_cpu[:, 1], Y_cpu[:, 0], minus_grad[:, 1], minus_grad[:, 0], angles='xy', scale_units='xy', scale=1)
                if target_name == 'circles':
                    plt.ylim([-1.0, 1.0])
                    plt.xlim([-2.0, 0.5])
                plt.gca().set_aspect('equal')
                plt.axis('off')
                plt.savefig(folder_name + img_name, dpi=300, bbox_inches='tight')
                plt.close()
            if d == 3:
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection="3d")
                fig.add_axes(ax)
                ax.view_init(azim=-66, elev=12)
                ax.scatter(target[:, 0], target[:, 1], target[:, 2], c='orange', s=2)
                ax.scatter(Y_cpu[:, 0], Y_cpu[:, 1], Y_cpu[:, 2], 'b.', s=2)
                plt.savefig(folder_name + img_name, dpi=300, bbox_inches='tight')
                plt.close()

        # construct kernel matrix
        kxy = kern(X[:, None, :], Y[None, :, :], s)
        kyy = kern(Y[:, None, :], Y[None, :, :], s)
        upper_row = torch.cat((kxx, kxy), dim=1)
        lower_row = torch.cat((kxy.t(), kyy), dim=1)
        K = torch.cat((upper_row, lower_row), dim=0)
        K = K.cpu()
        K = K.numpy()

        # calculate 1/2 MMD(X, Y)^2 and W2 metric between particles and target
        MMD[n] = (0.5 * (kxx.sum() / N ** 2 + kyy.sum() / M ** 2 - 2 * kxy.sum() / (N * M))).item()
        if compute_W2:
            M2 = ot.dist(X, Y, metric='sqeuclidean')
            W2[n] = ot.emd2(a, b, M2)

        # annealing
        if annealing and div_reces not in [0.0, float('inf')]:
            lower_bd_lambd = (2 * torch.sqrt(2*MMD[n]) * B / div_reces).item()
            lower_bds_lambd.append(lower_bd_lambd)
            if not (lambd > lower_bd_lambd):
                print("Condition is not fulfilled")
            if annealing_factor > 0 and n in [5e3, 1e4, 2e4]:
                lambd /= annealing_factor
                if verbose: print(f"new lambda = {lambd}")
            elif annealing_factor == 0 and lambd > 1e-2:
                lambd = lower_bd_lambd + 1e-4

        lambdas.append(lambd)

        # we first the simplified primal objectives for the case that div_reces = float('inf')
        def primal_objective(q):
            convex_term = np.sum(div(q, alpha))
            tilde_q = np.concatenate((q, - np.ones(N)))
            quadratic_term = tilde_q.T @ K @ tilde_q
            return 1/N * convex_term + 1/(2 * lambd * N * N) * quadratic_term

        def primal_jacobian(q):
            convex_term = div_der(q, alpha)
            tilde_q = np.concatenate((q, - np.ones(N)))
            linear_term = upper_row.cpu().numpy() @ tilde_q
            return 1/N * convex_term + 1/(lambd * N * N) * linear_term
        '''
        def primal_KALE_objective(q):
            convex_term = np.sum(div(q, 1))
            tilde_q = np.concatenate((q, - np.ones(N)))
            quadratic_term = tilde_q.T @ K @ tilde_q
            return 1/N * convex_term + 1/(2 * lambd * N * N) * quadratic_term

        def primal_KALE_jacobian(q):
            convex_term = div_der(q, 1)
            tilde_q = np.concatenate((q, - np.ones(N)))
            linear_term = upper_row.cpu().numpy() @ tilde_q
            return 1/N * convex_term + 1/(lambd * N * N) * linear_term
        '''

        # now the primal objective for div_reces < float('inf')
        def primal_objective_fin_rec(q):
            convex_term = 1/M * np.sum(div(q[:M], alpha))
            linear_term = div_reces * (1 + 1/M * np.sum(q[M:]))
            quadratic_term = 1/(2 * lambd * M * M) * q.T @ K @ q
            return convex_term + linear_term + quadratic_term

        def primal_jacobian_fin_rec(q):
            convex_term = div_der(q[:M], alpha)
            constnt_term = div_reces * np.ones(N)
            joint_term = 1/M * np.concatenate((convex_term, constnt_term))
            linear_term = K @ q
            return joint_term + 1/(lambd * M * M) * linear_term

        # this is minus the value of the objective
        def dual_objective(b):
            h = K @ b
            c1 = np.concatenate((div_conj(h[:M], alpha), - h[M:]))
            c3 = b.T @ h
            return 1/N * np.sum(c1) + lambd/2 * c3

        def dual_jacobian(b):
            h = K @ b
            x = np.concatenate((div_conj_der(h[:M], alpha), - np.ones(N)), axis=0)
            return 1/N * K @ x + lambd * h

        if n > 0:  # warm start
            warm_start_q = q_np  # take solution from last iteration
            if dual:
                if div_reces != float('inf'):
                    warm_start_b = - 1/(lambd*M) * q_np
                else:
                    warm_start_b = 1/(lambd*N) * np.concatenate((- q_np, np.ones(N)))
        else: # initial values
            if div_reces != float('inf'):
                warm_start_q = 1/1000*np.ones(N + M)
                if dual:
                    warm_start_b = - 1/(lambd * M) * warm_start_q
            else:
                warm_start_q = 1/1000*np.ones(N)
                if dual:
                    warm_start_b = 1/(lambd*N) * np.concatenate((- warm_start_q, np.ones(N)))

        optimizer_kwargs = dict(
            m=100,
            factr=100,
            pgtol=1e-7,
            iprint=0,
            maxiter=120,
            disp=0,
        )
        if div_reces != float('inf'):
            q_np, prim_value, _ = sp.optimize.fmin_l_bfgs_b(
                primal_objective_fin_rec,
                warm_start_q,
                fprime=primal_jacobian_fin_rec,
                bounds=[(0, None) for _ in range(M)] + [(-M/N, None) for _ in range(N)],
                **optimizer_kwargs)
        else:
            q_np, prim_value, _ = sp.optimize.fmin_l_bfgs_b(
                primal_objective,
                warm_start_q,
                fprime=primal_jacobian,
                bounds=[(0, None) for _ in range(N)],
                **optimizer_kwargs)
        '''
        if compute_KALE:
            _, prim_value_KALE, _ = sp.optimize.fmin_l_bfgs_b(
                primal_KALE_objective,
                warm_start_q,
                fprime=primal_KALE_jacobian,
                bounds=[(0, None) for _ in range(N + M)],
                **optimizer_kwargs)
            KALE_values[n] = prim_value_KALE
        '''
        func_values.append(prim_value)

        if dual:
            b_np, minus_dual_value, _ = sp.optimize.fmin_l_bfgs_b(dual_objective, warm_start_b, fprime = dual_jacobian, **opt_kwargs)     
            dual_values.append(-minus_dual_value)
            if plot and save_opts and not n % 1e5:
                torch.save(torch.from_numpy(b_np), f'{folder_name}/b_at_{n}.pt')

            if div_reces != float('inf'):
                pseudo_dual_value = - dual_objective(- 1/(lambd * M) * q_np)
            else:
                pseudo_dual_value = - dual_objective(1/(lambd * N) * np.concatenate((-q_np, np.ones(N))))

            pseudo_dual_values.append(pseudo_dual_value)
            pseudo_duality_gap = np.abs(prim_value - pseudo_dual_value)
            pseudo_duality_gaps.append(pseudo_duality_gap)
            relative_pseudo_duality_gap = pseudo_duality_gap / np.min((np.abs(prim_value), np.abs(pseudo_dual_value)))
            relative_pseudo_duality_gaps.append(relative_pseudo_duality_gap)
            pseudo_gap_tol, relative_pseudo_gap_tol = 1e-2, 1e-2
            if pseudo_duality_gap > pseudo_gap_tol and verbose:
                  warn(f'Iteration {n}: pseudo-duality gap = {pseudo_duality_gap:.4f} > tolerance = {pseudo_gap_tol}.')
            if relative_pseudo_duality_gap > relative_pseudo_gap_tol and verbose:
                  warn(f'Iteration {n}: relative pseudo-duality gap = {relative_pseudo_duality_gap:.4f} > tolerance = {relative_pseudo_gap_tol}.')
    
            dual_value = - minus_dual_value
            duality_gap = np.abs(prim_value - dual_value)
            duality_gaps.append(duality_gap)
            relative_duality_gap = duality_gap / np.min((np.abs(prim_value), np.abs(dual_value)))
            relative_duality_gaps.append(relative_duality_gap)
            gap_tol, relative_gap_tol = 1e-2, 1e-2
            if duality_gap > gap_tol and verbose:
                warn(f'Iteration {n}: duality gap = {duality_gap:.4f} > tolerance = {gap_tol}.')
            if relative_duality_gap > relative_gap_tol and verbose:
                warn(f'Iteration {n}: relative duality gap = {relative_duality_gap:.4f} > tolerance = {relative_gap_tol}.')
        q_torch = torch.tensor(q_np, dtype=torch.float64, device=my_device)
        # one could also use torch.from_numpy here

        # save solution vector in every 100-th iteration (to conserve memory)
        if plot and save_opts and not n % 1e5:
            torch.save(q_torch, f'{folder_name}/q_at_{n}.pt')

        Z = torch.cat((X, Y))
        if div_reces != float('inf'):
            temp = q_torch.view(M+N, 1, 1) * kern_der(Y, Z, s)
        else:
            temp = - kern_der(Y, Y, s) + q_torch.view(N, 1, 1) * kern_der(Y, X, s)

        h_star_grad = - 1 / (lambd * M) * torch.sum(temp, dim=0)
        if plot and save_opts and not n % 1e5:
            torch.save(h_star_grad, f'{folder_name}/h_star_grad_at_{n}.pt')

        # don't save particle position in every iteration (conserves memory)
        if not n % 1e4 or n in 100*np.arange(1, 10):
            torch.save(Y, f'{folder_name}/Y_at_{n}.pt')
        Y -= step_size * h_star_grad
        
    suffix = f',{lambd},{step_size},{N},{kernel},{s},{max_time},{target_name}'
    torch.save(func_values, folder_name + f'/Reg_{divergence}-{alpha}_Div_value_timeline{suffix}.pt')
    torch.save(MMD, folder_name + f'/Reg_{divergence}-{alpha}_Div_MMD_timeline{suffix}.pt')
    if compute_W2:
        torch.save(W2, folder_name + f'/Reg_{divergence}-{alpha}_DivW2_timeline{suffix}.pt')
    if dual:
        torch.save(duality_gaps, folder_name + f'/Reg_{divergence}-{alpha}_Divergence_duality_gaps_timeline{suffix}.pt')
        torch.save(relative_duality_gaps, folder_name + f'/Reg_{divergence}-{alpha}_Divergence_rel_duality_gaps_timeline{suffix}.pt')
    torch.save(pseudo_duality_gaps, folder_name + f'/Reg_{divergence}-{alpha}_Divergence_pseudo_duality_gaps_timeline{suffix}.pt')
    torch.save(relative_pseudo_duality_gaps, folder_name + f'/Reg_{divergence}-{alpha}_Divergence_rel_pseudo_duality_gaps_timeline{suffix}.pt')       
    if timeline:
        # plot MMD, objective value, and W2 along the flow
        fig, ax = plt.subplots()
        plt.plot(MMD.cpu().numpy())
        plt.xlabel('iterations')
        plt.ylabel(r'$\frac{1}{2} d_{K}(\mu, \nu)^2$')
        plt.yscale('log')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8)))
        plt.savefig(folder_name + f'/{divergence}_MMD_timeline,{alpha},{lambd},{step_size},{kernel},{s}.png', dpi=300, bbox_inches='tight')
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
        plt.legend(frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.savefig(folder_name + f'/{divergence}_objective_timeline,{alpha},{lambd},{step_size},{kernel},{s}.png', dpi=300, bbox_inches='tight')
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
          plt.savefig(folder_name + f'/{divergence}_W2_timeline,{alpha},{lambd},{step_size},{kernel},{s}.png', dpi=300, bbox_inches='tight')
          plt.close()

        # plot pseudo and relative duality gaps
        fig, ax = plt.subplots()
        if not alpha == '':
            plt.plot(duality_gaps, label='duality gap')
            plt.plot(relative_duality_gaps, '-.', label='relative duality gap')
        plt.plot(pseudo_duality_gaps, ':', label='pseudo duality gap')
        plt.plot(relative_pseudo_duality_gaps, label='relative pseudo-duality gap')
        plt.axhline(y=1e-2, linestyle='--', color='gray', label='tolerance')
        plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8)))
        plt.yscale('log')
        plt.xlabel('iterations')
        plt.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.savefig(folder_name + f'/{divergence}_duality_gaps_timeline,{alpha},{lambd},{step_size},{kernel},{s}.png', dpi=300, bbox_inches='tight')
        plt.close()


        # lower bd on lambda
        fig, ax = plt.subplots()
        plt.plot(lambdas, label=r'$\lambda$')
        plt.plot(lower_bds_lambd, label=r'lower bound on $\lambda$')
        plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8)))
        plt.yscale('log')
        plt.xlabel('iterations')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend(frameon=False)
        plt.savefig(folder_name + f'/{divergence}_lambd_timeline,{alpha},{lambd},{step_size},{kernel},{s}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        
        func_values = torch.tensor(np.array(func_values))
        KALE_values = torch.tensor(np.array(KALE_values))

    return func_values, MMD, W2, KALE_values

MMD_reg_f_div_flow(alpha = 5, target_name = 'circles', N = 900, lambd=.01, verbose=True, s=.05, annealing=False, kern=imq, compute_W2=False, arrows=True)