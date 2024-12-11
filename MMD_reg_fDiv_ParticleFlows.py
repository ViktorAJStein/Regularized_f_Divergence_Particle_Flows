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
from line_profiler import LineProfiler

torch.set_default_dtype(torch.float64)  # set higher precision
my_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def MMD_reg_f_div_flow(
        a=2,  # divergence parameter
        s=.1,  # kernel parameter
        N=30,  # number of prior particles
        M=30,  # number of target particles
        lambd=.01,  # regularization
        step_size=.001,  # step size for Euler forward discretization
        max_time=1,  # maximal time horizon for simulation
        plot=True,  # plot particles along the evolution
        arrows=False,  # plots arrows at particles to show their gradients
        timeline=True,  # plots timeline of functional value along the flow
        kern=inv_log,  # kernel
        dual=False,  # decide whether to solve dual problem as well
        div=tsallis,  # entropy function
        target_name='bananas',  # name of the target measure nu
        verbose=True,  # decide whether to print warnings and information
        compute_W2=False,  # compute W2 dist of particles to target along flow
        save_opts=False,  # save minimizers and gradients along the flow
        compute_KALE=False,  # compute MMD-reg. KL-div. from particle to target
        st=42,  # random state for reproducibility
        annealing=False,  # decide wether to use the annealing heuristic
        annealing_factor=0,  # factor by which to divide lambda
        tight=False,  # decide whether to use the tight variational formulation
        line_search='Polyak', # use the polyak stepsize, which only needs on gradient computation  
        FFBS=True,  # decide whether to use fast FBS for the not-tight problem
        torch_LBFGS_B=False,  # decide whether to use the torch (and thus GPU) version of L-BFGS-B
        Polyak=False
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
    torch.save(target, folder_name + '/target.pt')
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
    if not tight and not torch_LBFGS_B:
        kyy_cpu = kyy.cpu().numpy()
        kyy_sum = kyy.sum()
    if compute_W2:
        a = torch.ones(N) / N
        b = a

    if tight:  # mirror descent parameters
        number_of_steps_mirror = 200
        averaging = False
        if line_search == 'armijo':
            tau, c = 1/2, 1/2  # search control parameters of Armijo search
        elif line_search == 'Polyak':
            delta, b, cc = torch.tensor([100.0], device=my_device).double(), 100.0, torch.tensor([0.5+1e-5], device=my_device).double()

    snapshots = 1e2*np.arange(1, 10)
    for n in range(iterations):
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
            row_sum_kyx_torch = kyx.sum(dim=1)  # tensor of shape (M, )
            sum_kxx = kxx.sum()
            kyx_sum = row_sum_kyx_torch.sum()
            kyy_sum = kyy.sum()
        if not tight or FFBS and not torch_LBFGS_B:
            kyx_cpu = kyx.cpu().numpy()
            row_sum_kyx_cpu = np.sum(kyx_cpu, axis=1)  # np.array of shape (M, )
            kxx_cpu = kxx.cpu().numpy()
            sum_kxx = kxx_cpu.sum()
            kyx_sum = row_sum_kyx_cpu.sum()
            # this should be avoided, since for large M, N,
            # this large matrix does not fit into memory
            upper_row = torch.cat((kyy, kyx), dim=1)
            lower_row = torch.cat((kyx.t(), kxx), dim=1)
            K_torch = torch.cat((upper_row, lower_row), dim=0)
            # upper_row_cpu = upper_row.cpu().numpy()
            K = K_torch.cpu()
            K = K.numpy()

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
        def primal_objective(q):
            convex_term = 1/M * np.sum(div.fnc(q, a))
            quadratic_term = q @ (kyy_cpu @ q) - 2 * (M / N) * (q.T @ row_sum_kyx_cpu)
            return convex_term + 1 / (2 * lambd * M * M) * quadratic_term

        def primal_jacobian(q):
            convex_term = 1/M * div.der(q, a)
            linear_term = kyy_cpu @ q - M / N * row_sum_kyx_cpu
            return convex_term + 1 / (lambd * M * M) * linear_term

        def primal_objective_torch(q):
            convex_term = 1/M * torch.sum(div_torch(q, a))
            quadratic_term = q @ (kyy @ q) - 2 * (M / N) * (q.t() @ row_sum_kyx_torch) 
            return convex_term + 1/(2 * lambd * M * M) * quadratic_term

        def primal_jacobian_torch(q):
            convex_term = 1/M * div_der_torch(q, a)
            linear_term = kyy @ q - M / N * row_sum_kyx_torch
            return convex_term + 1/(lambd * M * M) * linear_term

        '''
        def primal_KALE_objective(q):
            convex_term = np.sum(div.fnc(q, 1))
            quadratic_term = q @ (kyy_cpu @ q) - 2 * (M / N) * (q.T @ row_sum_kyx_cpu)
            return 1/M * convex_term + 1/(2 * lambd * M * M) * quadratic_term

        def primal_KALE_jacobian(q):
            convex_term = div.der(q, 1)
            linear_term = kyy_cpu @ q - M / N * row_sum_kyx_cpu
            return 1/M * convex_term + 1/(lambd * M * M) * linear_term
        '''

        def dual_objective_reduced(b):  # b.shape = (M, )
            term_1 = kyy_cpu @ b + 1 / (lambd * N) * row_sum_kyx_cpu
            term_2 = 1 / M * div.conj(term_1, a).sum()
            quadratic_term = lambd / 2 * b @ (kyy_cpu @ b)
            return term_2 + quadratic_term

        def dual_jacobian_reduced(b):
            term_1 = kyy_cpu @ b + 1 / (lambd * N) * row_sum_kyx_cpu
            term_2 = 1 / M * div.conj_der(term_1, a) + lambd * b
            return kyy_cpu @ term_2

        # primal objective for div.reces(a) < float('inf')
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
        '''
        # this is minus the value of the objective
        # this should be modified into dual_objective_fin_rec
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
        '''

        if not tight:
            if FFBS:
                t = torch.tensor([1.0], device=my_device).double()
                if n > 0:  # warm start
                    q = q_torch
                else:
                    if div.reces(a) != float('inf'):
                        q = 1/10 * torch.ones(M + N, device=my_device).double()
                    else:
                        q = 1/10 * torch.ones(M, device=my_device)
                max_iter_FFBS = 100
                if div.reces(a) != float('inf'):
                    K_norm = torch.norm(K_torch)
                else:
                    K_norm = torch.norm(kyy)
                for k in range(max_iter_FFBS):
                    q_old = q.clone().to(my_device).double()
                    t_next = 1/2 * (1 + torch.sqrt(1 + 4 * t**2))
                    y = q + (t - 1)/t_next * (q - q_old)
                    if div.reces(a) != float('inf'):
                        y_tmp = y - 1/K_norm * K @ y
                        q = torch.cat((
                        div.prox(y_tmp[:M+1], a, lambd * M / K_norm), y_tmp[M+1:] - div.reces(a) * lambd * M / K_norm
                        ))
                    else:
                        y_tmp = y - 1 / K_norm * kyy @ y + M / (N * K_norm) * row_sum_kyx_torch
                        q = torch.tensor([div.prox(j, a, lambd * M / K_norm) for j in y_tmp], dtype=torch.double, device=my_device)
                    if torch.norm(y_tmp) < 1e-3 and torch.norm(q - q_old) / torch.norm(q) < 1e-3:
                        if verbose:
                            print(f'Converged in {k + 1} iterations')
                            print(f'Iter_diff = {torch.norm(q - q_old).item()}, residual = {torch.norm(y_tmp).item()}')
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
                            warm_start_q = 1/1000 * torch.ones(N + M, device=my_device)
                        else:
                            warm_start_q = 1/1000*np.ones(N + M)
                        if dual:
                            warm_start_b = - 1/(lambd * M) * warm_start_q
                    else:
                        if torch_LBFGS_B:
                            warm_start_q = 1/1000 * torch.ones(M, device=my_device)
                        else:
                            warm_start_q = 1/100*np.ones(M)
                        if dual:
                            warm_start_b = - 1/(lambd*N) * 1/100*np.ones(M)

                optimizer_kwargs = dict(factr=10000000.0)  # this is the default value
                if div.reces(a) != float('inf'):
                    q_np, prim_value, _ = sp.optimize.fmin_l_bfgs_b(
                        primal_objective_fin_rec,
                        warm_start_q,
                        fprime=primal_jacobian_fin_rec,
                        bounds=[(0, None)] * M + [(-M/N, None)] * N,
                        **optimizer_kwargs)
                    prim_value += div.reces(a)
                else:
                    if torch_LBFGS_B:
                        low = torch.zeros(M, device=my_device)
                        high = torch.tensor([float('inf')] * M, device=my_device).double()

                        # with torch.autograd.profiler.profile() as prof:
                        q_torch, prim_value = my_L_BFGS_B(warm_start_q, primal_objective_torch, low, high)
                        # print(prof.key_averages().table(sort_by="cpu_time_total"))
                    else:
                        q_np, prim_value, _ = sp.optimize.fmin_l_bfgs_b(
                            primal_objective,
                            warm_start_q,
                            fprime=primal_jacobian,
                            bounds=[(0, None)] * M,
                            **optimizer_kwargs)

        func_values[n] = prim_value + 1 / (2 * lambd * N**2) * sum_kxx

        if dual:
            if div.reces(a) != float('inf'):
                b_np, minus_dual_value, _ = sp.optimize.fmin_l_bfgs_b(dual_objective, warm_start_b, fprime = dual_jacobian, **optimizer_kwargs)
                dual_values[n] = - minus_dual_value
            else:
                b_np, minus_dual_value, _ = sp.optimize.fmin_l_bfgs_b(dual_objective_reduced, warm_start_b, fprime = dual_jacobian_reduced, **optimizer_kwargs)
                dual_values[n] = - minus_dual_value

            if plot and save_opts and not n % 1e5:
                torch.save(torch.from_numpy(b_np), f'{folder_name}/b_at_{n}.pt')
            # dual gaps
            dual_value = dual_values[n] + 1 / (2 * lambd * N**2) * sum_kxx
            duality_gap = np.abs(prim_value + minus_dual_value)
            duality_gaps[n] = duality_gap
            relative_duality_gap = duality_gap / np.min((np.abs(prim_value), np.abs(dual_value)))
            relative_duality_gaps[n] = relative_duality_gap
            gap_tol, relative_gap_tol = 1e-2, 1e-2
            if duality_gap > gap_tol and verbose:
                print(f'Iter {n}: duality gap = {duality_gap:.4f} > tol = {gap_tol}.')
            if relative_duality_gap > relative_gap_tol and verbose:
                print(f'Iter {n}: relative duality gap = {relative_duality_gap:.4f} > tol = {relative_gap_tol}.')
        '''
        if not tight or not FFBS or not torch_LBFGS_B:
            q_np = q_torch.cpu().numpy()
            if div.reces(a) != float('inf'):
                pseudo_dual_value = - dual_objective(1/(lambd * N) * np.concatenate((q_np, - M / N * np.ones(N))))
            else:
                pseudo_dual_value = - dual_objective_reduced(- 1/(lambd * M) * q_np)

            # (relative) pseudo-duality gaps
            pseudo_dual_values[n] = pseudo_dual_value
            if isinstance(prim_value, torch.Tensor):
                pseudo_duality_gap = torch.abs(prim_value - pseudo_dual_value)
                relative_pseudo_duality_gap = pseudo_duality_gap / torch.min((torch.abs(prim_value), torch.abs(pseudo_dual_value)))
            elif isinstance(prim_value, np.ndarray):
                pseudo_duality_gap = np.abs(prim_value - pseudo_dual_value)
                relative_pseudo_duality_gap = pseudo_duality_gap / np.min((np.abs(prim_value), np.abs(pseudo_dual_value)))
            pseudo_duality_gaps[n] = pseudo_duality_gap
            relative_pseudo_duality_gaps[n] = relative_pseudo_duality_gap
            pseudo_gap_tol, relative_pseudo_gap_tol = 1e-2, 1e-2
            if pseudo_duality_gap > pseudo_gap_tol and verbose:
                  print(f'Iter {n}: pseudo-duality gap = {pseudo_duality_gap:.4f} > tolerance = {pseudo_gap_tol}.')
            if relative_pseudo_duality_gap > relative_pseudo_gap_tol and verbose:
                  print(f'Iter {n}: relative pseudo-duality gap = {relative_pseudo_duality_gap:.4f} > tolerance = {relative_pseudo_gap_tol}.')

            q_torch = torch.tensor(q_np, dtype=torch.float64, device=my_device)

        # save solution vector in every 10000-th iteration (to conserve memory)
        if plot and save_opts and not n % 1e5:
            torch.save(q_torch, f'{folder_name}/q_at_{n}.pt')
        '''
        Z = torch.cat((Y, X))
        if div.reces(a) != float('inf'):
            temp = q_torch.view(M+N, 1, 1) * kern_der(X, Z, s)
        else:
            qtilde = torch.cat((q_torch, - M / N * torch.ones(N, device=my_device)))
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
        torch.save(W2, folder_name + f'/Reg_{div.name}-{a}_DivW2_{suffix}.pt')
    if dual:
        torch.save(duality_gaps, folder_name + f'/Reg_{div.name}-{a}_duality_gaps_{suffix}.pt')
        torch.save(relative_duality_gaps, folder_name + f'/Reg_{div.name}-{a}_rel_duality_gaps_{suffix}.pt')
    torch.save(pseudo_duality_gaps, folder_name + f'/Reg_{div.name}-{a}__pseudo_duality_gaps_{suffix}.pt')
    torch.save(relative_pseudo_duality_gaps, folder_name + f'/Reg_{div.name}-{a}__rel_pseudo_duality_gaps_{suffix}.pt')       
    if timeline:  # plot MMD, objective value, and W2 along the flow
        suffix = f'timeline,{a},{lambd},{step_size},{kernel},{s}'
        plot_MMD(MMD.cpu().numpy(), f'/{div.name}_MMD_{suffix}.png', folder_name)
        if tight or torch_LBFGS_B:
            func_values_cpu = func_values.cpu().numpy()
        else:
            func_values_cpu = func_values
        plot_func_values(a, dual_values, pseudo_dual_values, func_values_cpu , lambd, f'/{div.name}_objective_{suffix}.png', folder_name)
        plot_lambdas(lambdas, lower_bds_lambd, f'/{div.name}_lambd_{suffix}.png', folder_name)
        if compute_W2:
          plot_W2(W2.cpu().numpy(), f'/{div.name}_W2_timeline_{suffix}.png')
        if not tight:
            plot_gaps(a, duality_gaps, relative_pseudo_duality_gaps, pseudo_duality_gaps, relative_duality_gaps, f'/{div.name}_duality_gaps_{suffix}.png', folder_name)

    return func_values, MMD, W2, KALE_values


MMD_reg_f_div_flow()

lp = LineProfiler()
lp_wrapper = lp(MMD_reg_f_div_flow)
lp_wrapper()
lp.print_stats()
