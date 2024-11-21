import os
from warnings import warn
import torch
import ot
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from plotting import *
from kernels import *
from adds import *
from entropies import *
from data_generation import *
from backtracking import *

torch.set_default_dtype(torch.float64)  # set higher precision
use_cuda = torch.cuda.is_available()  # shorthand
my_device = 'cuda' if use_cuda else 'cpu'


def MMD_reg_f_div_flow(
        a=3,  # divergence parameter
        s=.1,  # kernel parameter
        N=100,  # number of prior particles
        M=100,  # number of target particles
        lambd=.01,  # regularization
        step_size=.001,  # step size for Euler forward discretization
        max_time=1,  # maximal time horizon for simulation
        plot=True,  # plot particles along the evolution
        arrows=False,  # plots arrows at particles to show their gradients
        timeline=True,  # plots timeline of functional value along the flow
        kern=inv_log,  # kernel
        dual=False,  # decide whether to solve dual problem as well
        div=tsallis,  # entropy function
        target_name='GMM',  # name of the target measure nu
        verbose=True,  # decide whether to print warnings and information
        compute_W2=False,  # compute W2 dist of particles to target along flow
        save_opts=False,  # save minimizers and gradients along the flow
        compute_KALE=False,  # compute MMD-reg. KL-div. from particle to target
        st=42,  # random state for reproducibility
        annealing=False,  # decide wether to use the annealing heuristic
        annealing_factor=0,  # factor by which to divide lambda
        tight=True,  # decide whether to use the tight variational formulation
        armijo = False  # decide whether to use Armijo backtracking line search in EG
        ):
    '''
    @return:    func_value:    torch tensor of length iterations,
                               objective value along the flow
                MMD:           torch tensor of length iterations,
                               1/2 MMD^2 between particles along the flow
                W2:            torch tensor of length iterations,
                               W2 metric between particles along the flow
                # KALE_values:   torch tensor of length iterations,
                #                regularized KL divergence between particles and target along the flow
    '''

    iterations = int(max_time / step_size) + 1  # max number of iterations

    kern_der = globals().get(kern.__name__ + '_der')
    kernel = kern.__name__
    B = emb_const(kern, s)  # embedding constant H_K \hookrightarrow C_0
    divergence = div.__name__
    div_reces = rec_const(divergence, a)  # recession constant of entropy function
    div_conj = globals().get(div.__name__ + '_conj')  # convex conjugate of the entropy function
    div_conj_der = globals().get(div.__name__ + '_conj_der')  # derivative of div_conj
    div_der = globals().get(div.__name__ + '_der')  # derivative of div
    div_torch = globals().get(div.__name__ + '_torch')  # derivative of div
    div_der_torch = globals().get(div.__name__ + '_der_torch')  # derivative of div
    
    folder_name = f"{divergence},a={a},lambd={lambd},tau={step_size},{kernel},{s},{N},{M},{max_time},{target_name},state={st}_{annealing}={annealing_factor},tight={tight},armijo={armijo}"
    make_folder(folder_name)

    if verbose and B:
        print(f'Kernel is {kernel}, embedding constant is {round(B,2)}, recession constant is {div_reces}')

    target, prior = generate_prior_target(N, M, st, target_name) 
    torch.save(target, folder_name + f'/target.pt')
    X = prior.clone().to(my_device)  # samples of prior distribution, shape = N x d
    Y = target.to(my_device)  # samples of target measure, shape = M x d
    d = len(Y[0])  # dimension of the ambient space in which the particles live

    func_values = np.zeros(iterations)  # objective value during the algorithm
    # KALE_values = torch.zeros(iterations)
    dual_values = np.zeros(iterations)
    lambdas = np.zeros(iterations)  # regularization parametre during the algorithm (relevant for annealing) 
    pseudo_dual_values = np.zeros(iterations)
    MMD = torch.zeros(iterations)  # 1/2 mmd(X, Y)^2 during the algorithm
    W2 = torch.zeros(iterations)
    duality_gaps = np.zeros(iterations)
    pseudo_duality_gaps = np.zeros(iterations)
    relative_duality_gaps = np.zeros(iterations)
    relative_pseudo_duality_gaps = np.zeros(iterations)
    lower_bds_lambd = np.zeros(iterations)

    kyy = kern(Y[:, None, :], Y[None, :, :], s)
    if not tight:
        kyy_cpu = kyy.cpu().numpy()
    if compute_W2:
        a, b = torch.ones(N) / N, torch.ones(N) / N

    snapshots = 1e2*np.arange(1, 10)
    for n in range(iterations):
        # plot the particles
        if plot and not n % 1000 or n in snapshots:
            if annealing:
                img_name = f'/Reg_{divergence}{a}flow,annealing,tau={step_size},{kernel},{s},{N},{M},{max_time},{target_name}-{n}.png'
            else:
                img_name = f'/Reg_{divergence}{a}flow,lambd={lambd},tau={step_size},{kernel},{s},{N},{M},{max_time},{target_name}-{n}.png'

            X_cpu = X.cpu()
            if d == 2:
                plt.figure()
                plt.plot(target[:, 1], target[:, 0], '.', c='orange', ms=2)
                plt.plot(X_cpu[:, 1], X_cpu[:, 0], 'b.', ms=2)
                if arrows and n > 0:
                    minus_grad = - h_star_grad.cpu()
                    plt.quiver(X_cpu[:, 1], X_cpu[:, 0], minus_grad[:, 1], minus_grad[:, 0], angles='xy', scale_units='xy', scale=1)
                if target_name == 'circles':
                    plt.ylim([-1.0, 1.0])
                    plt.xlim([-2.0, 0.5])
                if target_name == 'bananas':
                    plt.ylim([-7.5, 7.5])
                    plt.xlim([-5.0, 10.0])
                if target_name == 'four_wells':
                    plt.xlim([-2.5, 7.0])
                    plt.ylim([-2.5, 7.0])
                plt.gca().set_aspect('equal')
                plt.axis('off')
                plt.savefig(folder_name + img_name, dpi=300, bbox_inches='tight')
                plt.close()
            elif d == 3:
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection="3d")
                fig.add_axes(ax)
                ax.view_init(azim=-66, elev=12)
                ax.scatter(target[:, 0], target[:, 1], target[:, 2], c='orange', s=2)
                ax.scatter(X_cpu[:, 0], X_cpu[:, 1], X_cpu[:, 2], 'b.', s=2)
                plt.savefig(folder_name + img_name, dpi=300, bbox_inches='tight')
                plt.close()
        

        # construct kernel matrix
        kyx = kern(X[None, :, :], Y[:, None, :], s)
        kxx = kern(X[:, None, :], X[None, :, :], s)
        if tight:
            row_sum_kyx_torch = kyx.sum(dim=1) # tensor of shape (M, )
            sum_kxx = kxx.sum()
        
        if not tight:
            kyx_cpu = kyx.cpu().numpy()
            row_sum_kyx_cpu = np.sum(kyx_cpu, axis=1)  # np.array of shape (M, )
            kxx_cpu = kxx.cpu().numpy()
            sum_kxx_cpu = kxx_cpu.sum()
            # this should be avoided, since for large M, N, this large matrix does not fit into memory
            upper_row = torch.cat((kyy, kyx), dim=1)
            lower_row = torch.cat((kyx.t(), kxx), dim=1)
            K_torch = torch.cat((upper_row, lower_row), dim=0)
            upper_row_cpu = upper_row.cpu().numpy()
            K = K_torch.cpu()
            K = K.numpy()

        # calculate 1/2 MMD(X, Y)^2 and W2 metric between particles and target
        MMD[n] = (0.5 * (kyy.sum() / N ** 2 + kxx.sum() / M ** 2
                         - 2 * kyx.sum() / (N * M))).item()
        if compute_W2:
            M2 = ot.dist(Y, X, metric='sqeuclidean')
            W2[n] = ot.emd2(a, b, M2)

        # annealing
        if annealing and div_reces not in [0.0, float('inf')]:
            lower_bd_lambd = (2 * torch.sqrt(2*MMD[n]) * B / div_reces).item()
            lower_bds_lambd[n] = lower_bd_lambd
            if not (lambd > lower_bd_lambd):
                print("Condition is not fulfilled")
            if annealing_factor > 0 and n in [5e3, 1e4, 2e4]:
                lambd /= annealing_factor
                if verbose:
                    print(f"new lambda = {lambd}")
            elif annealing_factor == 0 and lambd > 1e-2:
                lambd = lower_bd_lambd + 1e-4

        lambdas[n] = lambd

        # first: reduced objectives for the case div_reces = float('inf')
        # TODO: reduce this objective
        def primal_objective(q):
            convex_term = 1/M * np.sum(div(q, a))
            quadratic_term = q.T @ kyy_cpu @ q - 2 * (M / N) * (q.T @ row_sum_kyx_cpu)
            return convex_term + 1/(2 * lambd * M * M) * quadratic_term

        def primal_jacobian(q):
            convex_term = 1/M * div_der(q, a)
            linear_term = kyy_cpu @ q - M / N * row_sum_kyx_cpu
            return convex_term + 1/(lambd * M * M) * linear_term

        def primal_objective_torch(q):
            convex_term = 1/M * torch.sum(div_torch(q, a))
            quadratic_term = q.t() @ kyy @ q - 2 * (M / N) * (q.t() @ row_sum_kyx_torch)
            return convex_term + 1/(2 * lambd * M * M) * quadratic_term

        def primal_jacobian_torch(q):
            convex_term = 1/M * div_der_torch(q, a)
            linear_term = kyy @ q - M / N * row_sum_kyx_torch
            return convex_term + 1/(lambd * M * M) * linear_term


        def primal_KALE_objective(q):
            convex_term = np.sum(div(q, 1))
            quadratic_term = quadratic_term = q.T @ kyy_cpu @ q - 2 * (M / N) * (q.T @ row_sum_kyx_cpu)
            return 1/M * convex_term + 1/(2 * lambd * M * M) * quadratic_term

        def primal_KALE_jacobian(q):
            convex_term = div_der(q, 1)
            linear_term = kyy_cpu @ q - M / N * row_sum_kyx_cpu
            return 1/M * convex_term + 1/(lambd * M * M) * linear_term

        def dual_objective_reduced(b):  # b.shape = (M, )
            convex_term = - 1 / M * div_conj(kyy_cpu @ b + 1 / (lambd * N) * row_sum_kyx_cpu, a).sum()
            quadratic_term = - lambd / 2 * b.T @ kyy_cpu @ b
            return convex_term + quadratic_term

        def dual_jacobian_reduced(b):
            convex_term = -1 / M * kyy_cpu @ div_conj_der(kyy_cpu @ b + 1 / (lambd * N) * row_sum_kyx_cpu, a)
            linear_term = - lambd * kyy_cpu @ b
            return convex_term + linear_term

        # now the primal objective for div_reces < float('inf')
        def primal_objective_fin_rec(q):
            convex_term = 1/M * np.sum(div(q[:M], a))
            linear_term = div_reces / M * np.sum(q[M:]))
            quadratic_term = 1/(2 * lambd * M * M) * q.T @ K @ q
            return convex_term + linear_term + quadratic_term

        def primal_jacobian_fin_rec(q):
            convex_term = div_der(q[:M], a)
            joint_term = 1/M * np.concatenate((convex_term, div_reces * np.ones(N)))
            linear_term = K @ q
            return joint_term + 1/(lambd * M * M) * linear_term

        # this is minus the value of the objective
        def dual_objective(b):
            h = K @ b
            c1 = np.concatenate((div_conj(h[:M], a), - h[M:]))
            c3 = b.T @ h
            return 1/N * np.sum(c1) + lambd/2 * c3

        def dual_jacobian(b):
            h = K @ b
            x = np.concatenate((div_conj_der(h[:M], a), - np.ones(N)), axis=0)
            return 1/N * K @ x + lambd * h

        if not tight:
            if n > 0:  # warm start
                warm_start_q = q_np  # take solution from last iteration
                if dual:
                    if div_reces != float('inf'):
                        warm_start_b = - 1/(lambd*M) * q_np
                    else:
                        warm_start_b = - 1/(lambd*N) * q_np
            else:  # initial values
                if div_reces != float('inf'):
                    warm_start_q = 1/1000*np.ones(N + M)
                    if dual:
                        warm_start_b = - 1/(lambd * M) * warm_start_q
                else:
                    warm_start_q = 1/1000*np.ones(M)
                    if dual:
                        warm_start_b = - 1/(lambd*N) * 1/1000*np.ones(M)
    
            optimizer_kwargs = dict(disp=0)
            if div_reces != float('inf'):
                q_np, prim_value, _ = sp.optimize.fmin_l_bfgs_b(
                    primal_objective_fin_rec,
                    warm_start_q,
                    fprime=primal_jacobian_fin_rec,
                    bounds=[(0, None) for _ in range(M)] + [(-M/N, None) for _ in range(N)],
                    **optimizer_kwargs)
                prim_value += div_reces
            else:
                q_np, prim_value, _ = sp.optimize.fmin_l_bfgs_b(
                    primal_objective,
                    warm_start_q,
                    fprime=primal_jacobian,
                    bounds=[(0, None) for _ in range(M)],
                    **optimizer_kwargs)
                prim_value += (M**2 / N**2) * sum_kxx_cpu
        else:
            number_of_steps_mirror = 80
            # accelerated mirror descent on M * unit simplex in R^M
            # with Armijo line search for choosing step size
            # todo: add restart
            q = torch.ones(M, device=my_device)  # initial vector
            q_end = q
            eta = .01  # fixed step size (guaranteed convergence if objective is L-smooth)
            averaging = False
            # r = 3
            tau, c = 1/2, 1/2  # search control parameters of Armijo search
            for k in range(number_of_steps_mirror):
                # p = r / (r + l) * q + (1 - r / (r + l)) * p
                ## Armijo
                if armijo:
                    eta = np.log(k+2)/(k+2)  # initial guess. Alternative: 1/np.sqrt(k+1) ?
                    p = primal_jacobian_torch(q)
                    t, fx = - c*torch.dot(p, q), primal_objective_torch(q)
                    eta = armijo_search(primal_objective_torch, q, eta, p, fx, t, tau)
                q_prev = q
                q = q * torch.exp(- eta * primal_jacobian_torch(q))
                q /= 1/M * torch.sum(q)
                res = torch.norm(primal_jacobian_torch(q)).item() / torch.norm(primal_jacobian_torch(torch.ones(M, device=my_device))).item()
                iter_diff = torch.norm(q - q_prev).item()
                if res < 1e-6 or iter_diff < 1e-6:  # Termination condition
                    if verbose: print(f"Converged in {k + 1} iterations.")
                    break
                else:
                    if verbose: print("Maximum iterations reached without convergence.")
                # p = TODO
                q_end += q
                # gradient restart of speed restart
                # if torch.dot(q_new - q_old, primal_jacobian_torch(q_old)) > 0 or torch.linalg.norm(q_new - q_old) < torch.linalg.norm(q_old - q_oldold):
                #     tilde_z = x
                #     l = 0  # reset momentum to 0
            if averaging:
                q_end /= number_of_steps_mirror
                q_torch, prim_value = q, primal_objective_torch(q) + (M**2 / N**2) * sum_kxx
            else:
                q_torch, prim_value = q, primal_objective_torch(q) + (M**2 / N**2) * sum_kxx
                      
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
        func_values[n] = prim_value

        if dual:
            if div_reces != float('inf'):
                b_np, minus_dual_value, _ = sp.optimize.fmin_l_bfgs_b(dual_objective, warm_start_b, fprime = dual_jacobian, **optimizer_kwargs)
                dual_values[n]= - minus_dual_value
            else:
                b_np, minus_dual_value, _ = sp.optimize.fmin_l_bfgs_b(dual_objective_reduced, warm_start_b, fprime = dual_jacobian_reduced, **optimizer_kwargs)
                minus_dual_value += 1 / (2 * lambd * N * N) * kxx_cpu.sum()
                dual_values[n]= - minus_dual_value

            if plot and save_opts and not n % 1e5:
                torch.save(torch.from_numpy(b_np), f'{folder_name}/b_at_{n}.pt')
            # dual gaps
            dual_value = - minus_dual_value
            duality_gap = np.abs(prim_value - dual_value)
            duality_gaps[n] = duality_gap
            relative_duality_gap = duality_gap / np.min((np.abs(prim_value), np.abs(dual_value)))
            relative_duality_gaps[n] = relative_duality_gap
            gap_tol, relative_gap_tol = 1e-2, 1e-2
            if duality_gap > gap_tol and verbose:
                warn(f'Iteration {n}: duality gap = {duality_gap:.4f} > tolerance = {gap_tol}.')
            if relative_duality_gap > relative_gap_tol and verbose:
                warn(f'Iteration {n}: relative duality gap = {relative_duality_gap:.4f} > tolerance = {relative_gap_tol}.')

        if not tight:
            if div_reces != float('inf'):
                pseudo_dual_value = - dual_objective(- 1/(lambd * M) * q_np)
            else:
                pseudo_dual_value = dual_objective(1/(lambd * N) * np.concatenate((q_np, - M / N * np.ones(N))))

            # (relative) pseudo-duality gaps
            pseudo_dual_values[n] = pseudo_dual_value
            pseudo_duality_gap = np.abs(prim_value - pseudo_dual_value)
            pseudo_duality_gaps[n] = pseudo_duality_gap
            relative_pseudo_duality_gap = pseudo_duality_gap / np.min((np.abs(prim_value), np.abs(pseudo_dual_value)))
            relative_pseudo_duality_gaps[n] = relative_pseudo_duality_gap
            pseudo_gap_tol, relative_pseudo_gap_tol = 1e-2, 1e-2
            if pseudo_duality_gap > pseudo_gap_tol and verbose:
                  warn(f'Iteration {n}: pseudo-duality gap = {pseudo_duality_gap:.4f} > tolerance = {pseudo_gap_tol}.')
            if relative_pseudo_duality_gap > relative_pseudo_gap_tol and verbose:
                  warn(f'Iteration {n}: relative pseudo-duality gap = {relative_pseudo_duality_gap:.4f} > tolerance = {relative_pseudo_gap_tol}.')

            q_torch = torch.tensor(q_np, dtype=torch.float64, device=my_device)

        # save solution vector in every 10000-th iteration (to conserve memory)
        if plot and save_opts and not n % 1e5:
            torch.save(q_torch, f'{folder_name}/q_at_{n}.pt')

        Z = torch.cat((Y, X))
        if div_reces != float('inf'):
            temp = q_torch.view(M+N, 1, 1) * kern_der(X, Z, s)
        else:
            qtilde = torch.cat( (q_torch, - M / N * torch.ones(N, device=my_device)) )
            temp = qtilde.view(M+N, 1, 1) * kern_der(X, Z, s)

        h_star_grad = - 1 / (lambd * M) * torch.sum(temp, dim=0)

        if plot and save_opts and not n % 1e5:
            torch.save(h_star_grad, f'{folder_name}/h_star_grad_at_{n}.pt')

        # don't save particle position in every iteration (conserves memory)
        if not n % 1e4 or n in 100*np.arange(1, 10):
            torch.save(Y, f'{folder_name}/Y_at_{n}.pt')
        X -= step_size * h_star_grad

    suffix = f',{lambd},{step_size},{N},{M},{kernel},{s},{max_time},{target_name}'
    torch.save(func_values, folder_name + f'/Reg_{divergence}-{a}_Div_value_timeline{suffix}.pt')
    torch.save(MMD, folder_name + f'/Reg_{divergence}-{a}_Div_MMD_timeline{suffix}.pt')
    if compute_W2:
        torch.save(W2, folder_name + f'/Reg_{divergence}-{a}_DivW2_timeline{suffix}.pt')
    if dual:
        torch.save(duality_gaps, folder_name + f'/Reg_{divergence}-{a}_Divergence_duality_gaps_timeline{suffix}.pt')
        torch.save(relative_duality_gaps, folder_name + f'/Reg_{divergence}-{a}_Divergence_rel_duality_gaps_timeline{suffix}.pt')
    torch.save(pseudo_duality_gaps, folder_name + f'/Reg_{divergence}-{a}_Divergence_pseudo_duality_gaps_timeline{suffix}.pt')
    torch.save(relative_pseudo_duality_gaps, folder_name + f'/Reg_{divergence}-{a}_Divergence_rel_pseudo_duality_gaps_timeline{suffix}.pt')       
    if timeline: # plot MMD, objective value, and W2 along the flow
        suffix = f'{a},{lambd},{step_size},{kernel},{s}'
        plot_MMD(MMD.cpu().numpy(), f'/{divergence}_MMD_timeline,{suffix}.png', folder_name)
        plot_func_values(a, dual_values, pseudo_dual_values, func_values, lambd, f'/{divergence}_objective_timeline,{suffix}.png', folder_name)
        plot_lambdas(lambdas, lower_bds_lambd, f'/{divergence}_lambd_timeline,{suffix}.png', folder_name)
        if compute_W2:
          plot_W2(W2.cpu().numpy(), f'/{divergence}_W2_timeline,{suffix}.png')
        if not tight:
            plots_gaps(a, duality_gaps, pseudo_duality_gaps, relative_duality_gaps, f'/{divergence}_duality_gaps_timeline,{suffix}.png', folder_name)

    return torch.tensor(func_values), MMD, W2 #, KALE_values

MMD_reg_f_div_flow()
