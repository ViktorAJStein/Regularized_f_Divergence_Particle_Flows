import os
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
from my_lbfgsb import *

torch.set_default_dtype(torch.float64)  # set higher precision
my_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def MMD_reg_f_div_flow(
        a=4,  # divergence parameter
        s=.1,  # kernel parameter
        N=100,  # number of prior particles
        M=100,  # number of target particles
        lambd=.1,  # regularization
        step_size=.001,  # step size for Euler forward discretization
        max_time=1,  # maximal time horizon for simulation
        plot=True,  # plot particles along the evolution
        arrows=False,  # plots arrows at particles to show their gradients
        timeline=True,  # plots timeline of functional value along the flow
        kern=imq,  # kernel
        dual=False,  # decide whether to solve dual problem as well
        div=tsallis,  # entropy function
        target_name='bananas',  # name of the target measure nu
        verbose=False,  # decide whether to print warnings and information
        compute_W2=False,  # compute W2 dist of particles to target along flow
        save_opts=False,  # save minimizers and gradients along the flow
        compute_KALE=False,  # compute MMD-reg. KL-div. from particle to target
        st=42,  # random state for reproducibility
        annealing=False,  # decide wether to use the annealing heuristic
        annealing_factor=0,  # factor by which to divide lambda
        tight=True,  # decide whether to use the tight variational formulation
        line_search = 'Polyak', # use the polyak stepsize, which only needs on gradient computation  
        FFBS=False,  # decide whether to use fast FBS for the not-tight problem
        torch_LBFGS_B= False,  # decide whether to use the torch (and thus GPU) version of L-BFGS-B
        Polyak = True 
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
    if target_name == 'circles' and M == N and M % 3:
        M = (M // 3) * 3
        N = M

    iterations = int(max_time / step_size) + 1  # max number of iterations

    kern_der = globals().get(kern.__name__ + '_der')
    kernel = kern.__name__
    B = emb_const(kern, s)  # embedding constant H_K \hookrightarrow C_0 
    div_torch = globals().get(div.name + '_torch')  # derivative of div
    div_der_torch = globals().get(div.name + '_der_torch')  # derivative of div
    folder_name = f"{div.name},a={a},lambd={lambd},tau={step_size},{kernel},{s},{N},{M},{max_time},{target_name},state={st},{annealing}={annealing_factor},tight={tight},line_search={line_search},FFBS={FFBS}"
    make_folder(folder_name)

    if verbose and B:
        print(f'Kernel is {kernel}, embedding constant is {round(B,2)}, recession constant is {div.reces(a)}')

    target, prior = generate_prior_target(N, M, st, target_name) 
    torch.save(target, folder_name + f'/target.pt')
    X = prior.clone().to(my_device)  # samples of prior distribution, shape = N x d
    Y = target.to(my_device)  # samples of target measure, shape = M x d
    d = len(Y[0])  # dimension of the ambient space in which the particles live

    if tight or torch_LBFGS_B:
        func_values = torch.zeros(iterations, device=my_device)  # objective value during the algorithm
    else:
        func_values = np.zeros(iterations)
    KALE_values = torch.zeros(iterations, device=my_device)
    dual_values = np.zeros(iterations)
    lambdas = np.zeros(iterations)  # regularization parameter during the algorithm (relevant for annealing) 
    pseudo_dual_values = np.zeros(iterations)
    MMD = torch.zeros(iterations, device=my_device)  # 1/2 mmd(X, Y)^2 during the algorithm
    W2 = torch.zeros(iterations, device=my_device)
    duality_gaps = np.zeros(iterations)
    pseudo_duality_gaps = np.zeros(iterations)
    relative_duality_gaps = np.zeros(iterations)
    relative_pseudo_duality_gaps = np.zeros(iterations)
    lower_bds_lambd = np.zeros(iterations)

    kyy = kern(Y[:, None, :], Y[None, :, :], s)
    '''
    if not tight and not torch_LBFGS_B:
        kyy_cpu = kyy.cpu().numpy()
        kyy_sum = kyy.sum()
    '''
    if compute_W2:
        a = torch.ones(N) / N
        b = a
        
    if tight:  # mirror descent parameters
        number_of_steps_mirror = 200
        averaging = False
        if line_search == 'armijo':
            tau, c = 1/2, 1/2  # search control parameters of Armijo search
        elif line_search == 'Polyak':
            delta, B, cc = 100, 100, 1/2+1e-5

    snapshots = 1e2*np.arange(1, 10)
    for n in range(iterations):
        print(n)
        # plot the particles
        if plot and not n % 1000 or n in snapshots:
            if annealing:
                img_name = f'/Reg_{div.name}{a}flow,annealing,tau={step_size},{kernel},{s},{N},{M},{max_time},{target_name}-{n}.png'
            else:
                img_name = f'/Reg_{div.name}{a}flow,lambd={lambd},tau={step_size},{kernel},{s},{N},{M},{max_time},{target_name}-{n}.png'

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
        if tight or FFBS or torch_LBFGS_B:
            row_sum_kyx_torch = kyx.sum(dim=1) # tensor of shape (M, )
            sum_kxx = kxx.sum()
            kyx_sum = row_sum_kyx_torch.sum()
            kyy_sum = kyy.sum()
        '''
        if not tight or FFBS and not torch_LBFGS_B:
            kyx_cpu = kyx.cpu().numpy()
            row_sum_kyx_cpu = np.sum(kyx_cpu, axis=1)  # np.array of shape (M, )
            kxx_cpu = kxx.cpu().numpy()
            sum_kxx = kxx_cpu.sum()
            kyx_sum = row_sum_kyx_cpu.sum()
            # this should be avoided, since for large M, N, this large matrix does not fit into memory
            upper_row = torch.cat((kyy, kyx), dim=1)
            lower_row = torch.cat((kyx.t(), kxx), dim=1)
            K_torch = torch.cat((upper_row, lower_row), dim=0)
            upper_row_cpu = upper_row.cpu().numpy()
            K = K_torch.cpu()
            K = K.numpy()
        '''
        # calculate 1/2 MMD(X, Y)^2 and W2 metric between particles and target
        mmd = kyy_sum / N ** 2 + sum_kxx / M ** 2 - 2 * kyx_sum / (N * M)
        MMD[n] = mmd
        if compute_W2:
            M2 = ot.dist(Y, X, metric='sqeuclidean')
            W2[n] = ot.emd2(a, b, M2)

        # annealing
        if annealing and div.reces(a) not in [0.0, float('inf')]:
            lower_bds_lambd[n] = 2 * torch.sqrt(2*MMD[n]) * B / div.reces(a)
            if not (lambd > lower_bd_lambd):
                print("Condition is not fulfilled")
            if annealing_factor > 0 and n in [5e3, 1e4, 2e4]:
                lambd /= annealing_factor
                if verbose:
                    print(f"new lambda = {lambd}")
            elif annealing_factor == 0 and lambd > 1e-2:
                lambd = lower_bd_lambd + 1e-4

        lambdas[n] = lambd

        # first: reduced objectives for the case div.reces(a) = float('inf')
        # TODO: reduce this objective
        def primal_objective(q):
            convex_term = 1/M * np.sum(div.fnc(q, a))
            quadratic_term = q @ (kyy_cpu @ q) - 2 * (M / N) * (q.T @ row_sum_kyx_cpu)
            return convex_term + 1/(2 * lambd * M * M) * quadratic_term

        def primal_jacobian(q):
            convex_term = 1/M * div.der(q, a)
            linear_term = kyy_cpu @ q - M / N * row_sum_kyx_cpu
            return convex_term + 1/(lambd * M * M) * linear_term

        def primal_objective_torch(q):
            convex_term = 1/M * torch.sum(div_torch(q, a))
            quadratic_term = q @ (kyy @ q) - 2 * (M / N) * (q.t() @ row_sum_kyx_torch)
            return convex_term + 1/(2 * lambd * M * M) * quadratic_term

        def primal_jacobian_torch(q):
            convex_term = 1/M * div_der_torch(q, a)
            linear_term = kyy @ q - M / N * row_sum_kyx_torch
            return convex_term + 1/(lambd * M * M) * linear_term


        def primal_KALE_objective(q):
            convex_term = np.sum(div.fnc(q, 1))
            quadratic_term = quadratic_term = q @ (kyy_cpu @ q) - 2 * (M / N) * (q.T @ row_sum_kyx_cpu)
            return 1/M * convex_term + 1/(2 * lambd * M * M) * quadratic_term

        def primal_KALE_jacobian(q):
            convex_term = div.der(q, 1)
            linear_term = kyy_cpu @ q - M / N * row_sum_kyx_cpu
            return 1/M * convex_term + 1/(lambd * M * M) * linear_term

        def dual_objective_reduced(b):  # b.shape = (M, )
            convex_term = - 1 / M * div.conj(kyy_cpu @ b + 1 / (lambd * N) * row_sum_kyx_cpu, a).sum()
            quadratic_term = - lambd / 2 * b @ (kyy_cpu @ b)
            return convex_term + quadratic_term

        def dual_jacobian_reduced(b):
            convex_term = -1 / M * kyy_cpu @ div_conj_der(kyy_cpu @ b + 1 / (lambd * N) * row_sum_kyx_cpu, a)
            linear_term = - lambd * kyy_cpu @ b
            return convex_term + linear_term

        # now the primal objective for div.reces(a) < float('inf')
        def primal_objective_fin_rec(q):
            convex_term = 1/M * np.sum(div.fnc(q[:M], a))
            linear_term = div.reces(a) / M * np.sum(q[M:])
            quadratic_term = 1/(2 * lambd * M * M) * q @ (K @ q)
            return convex_term + linear_term + quadratic_term

        def primal_jacobian_fin_rec(q):
            convex_term = div.der(q[:M], a)
            joint_term = 1/M * np.concatenate((convex_term, div.reces(a) * np.ones(N)))
            linear_term = K @ q
            return joint_term + 1/(lambd * M * M) * linear_term

        # this is minus the value of the objective
        def dual_objective(b):
            h = K @ b
            c1 = np.concatenate((div.conj(h[:M], a), - h[M:]))
            c3 = b.T @ h
            return 1/N * np.sum(c1) + lambd/2 * c3

        def dual_jacobian(b):
            h = K @ b
            x = np.concatenate((div_conj_der(h[:M], a), - np.ones(N)), axis=0)
            return 1/N * K @ x + lambd * h
            
        def dual_objective_torch(b):
            h = K_torch @ b
            c1 = torch.cat((div.conj(h[:M], a), - h[M:]))
            c3 = b.T @ h
            return 1/N * torch.sum(c1) + lambd/2 * c3
        
        def dual_jacobian_torch(b):
            h = K_torch @ b
            x = torch.cat((div.conj_der(h[:M], a), - torch.ones(N)), axis=0)
            return 1/N * K_torch @ x + lambd * h
            
        if not tight:
            if FFBS:
                t = torch.tensor([1], device=my_device)
                if n > 0:  # warm start
                    q, x = q_torch, q_torch
                else:
                    if div.reces(a) != float('inf'):
                        q, x = 1/10 * torch.ones(M + N, device=my_device), 1/10 * torch.ones(M + N, device=my_device)
                    else:
                        q, x = 1/10 * torch.ones(M, device=my_device), 1/10 * torch.ones(M, device=my_device)
                max_iter_FFBS = 100
                if div.reces(a) != float('inf'):
                    K_norm = torch.norm(K_torch)
                else:
                    K_norm = torch.norm(kyy)
                step_size_FFBS = 2*lambd*M*M / K_norm - 1e-4
                for k in range(max_iter_FFBS):
                    q_old = q.clone().to(my_device)
                    t_next = 1/2 * (1 + torch.sqrt(1 + 4 * t**2))
                    y = q + (t - 1)/t_next * (q - q_old)
                    if div.reces(a) != float('inf'):
                        grad_y = div.reces(a) / M * torch.concatenate((torch.ones(M), torch.zeros(N))) + 1/(lambd * M * M) * K_torch @ y
                        q = torch.concatenate((div.prox(y - step_size_FFBS * grad_y, a, step_size_FFBS), y[M+1:]))
                    else:
                        grad_y = 1/(lambd * M * M) * kyy @ y
                        print(div.prox(y - step_size_FFBS * grad_y, a, step_size_FFBS))
                        q = torch.tensor(div.prox(y - step_size_FFBS * grad_y, a, step_size_FFBS), device=my_device)
                    if torch.norm(grad_y) < 1e-3 and torch.norm(q - q_old) / torch.norm(q) < 1e-3:
                        if verbose:
                            print(f'Converged in {k + 1} iterations')
                            print(f'Iter_diff = {torch.norm(q - q_old).item()}, residual = {torch.norm(grad_y).item()}')
                        break
                    t = t_next
                prim_value = primal_objective_torch(q)
                q_torch = q
            else:
                if n > 0:  # warm start
                    if torch_LBFGS_B:
                        warm_start_q = q_torch
                    else:
                        warm_start_q = q_np  # take solution from last iteration
                    if dual:
                        if div.reces(a) != float('inf'):
                            warm_start_b = - 1/(lambd*M) * q_np
                        else:
                            warm_start_b = - 1/(lambd*N) * q_np
                else:  # initial values
                    if div.reces(a) != float('inf'):
                        if torch_LBFGS_B:
                            warm_start_q = 1/1000 * torch.ones(N + M, device = my_device)
                        else:
                            warm_start_q = 1/1000*np.ones(N + M)
                        if dual:
                            warm_start_b = - 1/(lambd * M) * warm_start_q
                    else:
                        if torch_LBFGS_B:
                            warm_start_q = 1/1000 * torch.ones(M, device = my_device)
                        else:
                            warm_start_q = 1/1000*np.ones(M)
                        if dual:
                            warm_start_b = - 1/(lambd*N) * 1/1000*np.ones(M)
        
                optimizer_kwargs = dict(disp=0)
                if div.reces(a) != float('inf'):
                    q_np, prim_value, _ = sp.optimize.fmin_l_bfgs_b(
                        primal_objective_fin_rec,
                        warm_start_q,
                        fprime=primal_jacobian_fin_rec,
                        bounds=[(0, None) for _ in range(M)] + [(-M/N, None) for _ in range(N)],
                        **optimizer_kwargs)
                    prim_value += div.reces(a)
                else:
                    if torch_LBFGS_B:
                        low = torch.zeros(M, device=my_device)
                        high = torch.tensor([float('inf')] * M, device=my_device)
                        
                        # with torch.autograd.profiler.profile() as prof:
                        q_torch, prim_value = my_L_BFGS_B(warm_start_q, primal_objective_torch, low, high)
                        print(q_torch, prim_value)
                        # print(prof.key_averages().table(sort_by="cpu_time_total"))
                        
                        prim_value += (M**2 / N**2) * sum_kxx
                    else: 
                        q_np, prim_value, _ = sp.optimize.fmin_l_bfgs_b(
                            primal_objective,
                            warm_start_q,
                            fprime=primal_jacobian,
                            bounds=[(0, None) for _ in range(M)],
                            **optimizer_kwargs)
                        prim_value += (M**2 / N**2) * sum_kxx
        else:
            # mirror descent on M * unit simplex in R^M
            # with differen line search methods for choosing step size
            # todo: add restart
            if n == 0:
                q = torch.ones(M, device=my_device)  # initial vector
                norm_factor = torch.norm(primal_objective_torch(torch.ones(M, device=my_device)))
            else:
                q = q_torch  # warm start
            q_end = q
            f_recs = torch.zeros(number_of_steps_mirror, device=my_device)
            for k in range(number_of_steps_mirror):
                fx = primal_objective_torch(q)
                if line_search == 'Polyak':
                    sig, l = 0, 0
                    f_recs[k] = fx.item()
                    f_rec = torch.min(f_recs[:k+1])
                    idx = torch.zeros(2*number_of_steps_mirror+1, device=my_device)
                    idx[0] = 1
                eta = np.log(k+2)/(k+2)  # initial guess
                if k == 0: 
                    p = primal_jacobian_torch(q)
                if line_search == 'armijo':
                    t = - c*torch.dot(p, q)
                    eta = armijo_search(primal_objective_torch, q, eta, p, fx, t, tau)
                elif line_search == 'Polyak':
                    if primal_objective_torch(q) <= f_recs[int(idx[l])] - 1/2*delta:
                        idx[l+1] = k
                        sig = 0
                        l += 1
                    elif sig > B:
                        idx[l+1] = k
                        sig = 0
                        delta /= 2
                        l += 1
                
                    fhat = f_recs[int(idx[l])] - delta
                    grad_norm = torch.norm(p)
                    eta = (fx - fhat)/(cc*grad_norm**2)
                    sig += eta*grad_norm
                else:  # use two-way backtracking search
                    eta = two_way_backtracking_line_search(primal_objective_torch, primal_jacobian_torch, q, -p, alpha_0=eta)
                q_prev = q
                q = q * torch.exp(- eta * p)
                q /= 1/M * torch.sum(q)
                p = primal_jacobian_torch(q)  # new gradient
                rel_res = torch.norm(p) / norm_factor
                iter_diff = torch.norm(q - q_prev)
                if rel_res < 1e-3 or iter_diff < 1e-3:  # Termination condition
                    if verbose:
                        print(f"Relative residual is {rel_res.item()}, iteration difference is {iter_diff.item()}")
                        print(f"Converged in {k + 1} iterations.")
                    break
                else:
                    if verbose:
                        print(f"Relative residual is {rel_res.item()}, iteration difference is {iter_diff.item()}")
                        print("Maximum iterations reached without convergence.")
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
            if div.reces(a) != float('inf'):
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
                print(f'Iteration {n}: duality gap = {duality_gap:.4f} > tolerance = {gap_tol}.')
            if relative_duality_gap > relative_gap_tol and verbose:
                print(f'Iteration {n}: relative duality gap = {relative_duality_gap:.4f} > tolerance = {relative_gap_tol}.')
        '''
        if not tight or not FFBS or not torch_LBFGS_B:
            if div.reces(a) != float('inf'):
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
                  print(f'Iteration {n}: pseudo-duality gap = {pseudo_duality_gap:.4f} > tolerance = {pseudo_gap_tol}.')
            if relative_pseudo_duality_gap > relative_pseudo_gap_tol and verbose:
                  print(f'Iteration {n}: relative pseudo-duality gap = {relative_pseudo_duality_gap:.4f} > tolerance = {relative_pseudo_gap_tol}.')

            q_torch = torch.tensor(q_np, dtype=torch.float64, device=my_device)
        '''
        # save solution vector in every 10000-th iteration (to conserve memory)
        if plot and save_opts and not n % 1e5:
            torch.save(q_torch, f'{folder_name}/q_at_{n}.pt')

        Z = torch.cat((Y, X))
        if div.reces(a) != float('inf'):
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

    suffix = f'timeline,{lambd},{step_size},{N},{M},{kernel},{s},{max_time},{target_name}'
    torch.save(func_values, folder_name + f'/Reg_{div.name}-{a}_Div_value_{suffix}.pt')
    torch.save(MMD, folder_name + f'/Reg_{div.name}-{a}_MMD_{suffix}.pt')
    if compute_W2:
        torch.save(W2, folder_name + f'/Reg_{div.name}-{a}_DivW2_timeline{suffix}.pt')
    if dual:
        torch.save(duality_gaps, folder_name + f'/Reg_{div.name}-{a}_duality_gaps_{suffix}.pt')
        torch.save(relative_duality_gaps, folder_name + f'/Reg_{div.name}-{a}_rel_duality_gaps_{suffix}.pt')
    torch.save(pseudo_duality_gaps, folder_name + f'/Reg_{div.name}-{a}__pseudo_duality_gaps_{suffix}.pt')
    torch.save(relative_pseudo_duality_gaps, folder_name + f'/Reg_{div.name}-{a}__rel_pseudo_duality_gaps_{suffix}.pt')       
    if timeline: # plot MMD, objective value, and W2 along the flow
        suffix = f'timeline,{a},{lambd},{step_size},{kernel},{s}'
        plot_MMD(MMD.cpu().numpy(), f'/{div.name}_MMD_timeline,{suffix}.png', folder_name)
        if tight or torch_LBFGS_B:
            func_values_cpu = func_values.cpu().numpy()
        else:
            func_values_cpu = func_values
        plot_func_values(a, dual_values, pseudo_dual_values, func_values_cpu , lambd, f'/{div.name}_objective_{suffix}.png', folder_name)
        plot_lambdas(lambdas, lower_bds_lambd, f'/{div.name}_lambd_timeline,{suffix}.png', folder_name)
        if compute_W2:
          plot_W2(W2.cpu().numpy(), f'/{div.name}_W2_timeline,{suffix}.png')
        if not tight:
            plot_gaps(a, duality_gaps, relative_pseudo_duality_gaps, pseudo_duality_gaps, relative_duality_gaps, f'/{div.name}_duality_gaps_{suffix}.png', folder_name)

    return func_values, MMD, W2 , KALE_values

MMD_reg_f_div_flow()

from line_profiler import LineProfiler
lp = LineProfiler()
lp_wrapper = lp(MMD_reg_f_div_flow)
lp_wrapper()
lp.print_stats()
