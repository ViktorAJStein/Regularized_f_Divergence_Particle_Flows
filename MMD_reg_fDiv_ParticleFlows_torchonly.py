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
from entropies_torchonly import *
from data_generation import *
from backtracking import *
# from torch.profiler import profile, record_function, ProfilerActivity
import time
from tqdm import tqdm

torch.set_default_dtype(torch.float64)  # set higher precision
my_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def MMD_reg_f_div_flow(
        a=2,  # divergence parameter
        s=.1,  # kernel parameter
        N=300,  # number of prior particles
        M=300,  # number of target particles
        lambd=.01,  # regularization
        step_size=.01,  # step size for Euler forward discretization
        max_time=10.0,  # maximal time horizon for simulation
        plot=True,  # plot particles along the evolution
        arrows=False,  # plots arrows at particles to show their gradients
        timeline=True,  # plots timeline of functional value along the flow
        kern=inv_log,  # kernel
        primal=True,  # decide whether to solve the primal problem
        dual=False,  # decide whether to solve dual problem
        div=tsa,  # entropy function
        target_name='bananas',  # name of the target measure nu
        verbose=True,  # decide whether to print warnings and information
        compute_W2=False,  # compute W2 dist of particles to target along flow
        save_opts=False,  # save minimizers and gradients along the flow
        st=42,  # random state for reproducibility
        annealing=False,  # decide wether to use the annealing heuristic
        annealing_factor=1,  # factor by which to divide lambda
        tight=False,  # decide whether to use the tight variational formulation
        line_search='armijo'  # stepsize for mirror descent if tight == True
        ):
    '''
    @return:    primal_values: torch tensor of length iterations,
                               objective value along the flow
                MMD:           torch tensor of length iterations,
                               1/2 MMD^2 between particles along the flow
                W2:            torch tensor of length iterations,
                               W2 metric between particles along the flow
    '''
    if not primal and not dual:
        raise Exception('Select either primal or problem to be solved')
    # First, some hyperparameters, like (relative) (pseudo-)duality gap tolerance
    start = time.time()

    gap_tol, relative_gap_tol = 1e-2, 1e-2
    pseudo_gap_tol, relative_pseudo_gap_tol = 1e-2, 1e-2
    max_iter_FFBS = 50
    
    if tight and div.reces(a) != float('Inf'):
        raise Exception('Tight variational formulation only available for entropy function with infinite recession constant')

    if compute_W2:
        a = torch.ones(N, device=my_device) / N
        b = a

    if tight:  # set mirror descent parameters
        number_of_steps_mirror = 75
        averaging = False
        if line_search == 'armijo':
            tau, c = 0.5, 0.5  # search control parameters of Armijo search
        elif line_search == 'Polyak':
            delta, b, cc = 1.0, 1.0, 0.5+1e-5
    '''
    if target_name == 'circles' and M == N and M % 3:
        M = (M // 3) * 3
        N = M
    '''

    # Now for the main algorithm
    iterations = int(max_time / step_size) + 1  # max number of iterations

    kern_der = globals().get(kern.__name__ + '_der')
    kernel = kern.__name__
    B = emb_const(kern, s)  # embedding constant H_K \hookrightarrow C_0
    if annealing:
        folder_name = f"{div.name},a={a},lambd={lambd},tau={step_size},{kernel},{s},{N},{M},{max_time},{target_name},state={st},primal={primal},dual={dual},annealing={annealing_factor},tight={tight},line_search={line_search}"
    else:
        folder_name = f"{div.name},a={a},lambd={lambd},tau={step_size},{kernel},{s},{N},{M},{max_time},{target_name},state={st},primal={primal},dual={dual},tight={tight},line_search={line_search}"
    make_folder(folder_name)

    if verbose:
        print(f'Divergence is {div.name}-{a}, kernel is {kernel}, recession constant is {div.reces(a)}')

    target, prior = generate_prior_target(N, M, st, target_name)
    torch.save(target, folder_name + '/target.pt')
    X = prior.clone().to(my_device)  # samples of prior distribution, shape = N x d
    Y = target.to(my_device)  # samples of target measure, shape = M x d
    d = len(Y[0])  # dimension of the ambient space in which the particles live

    primal_values = torch.zeros(iterations, device=my_device)  # objective value during the algorithm
    dual_values = torch.zeros(iterations, device=my_device)
    pseudo_primal_values = torch.zeros(iterations, device=my_device)  # objective value during the algorithm
    pseudo_dual_values = torch.zeros(iterations, device=my_device)
    duality_gaps = torch.zeros(iterations)
    relative_duality_gaps = torch.zeros(iterations)
    primal_pseudo_duality_gaps = torch.zeros(iterations)
    relative_primal_pseudo_duality_gaps = torch.zeros(iterations)
    dual_pseudo_duality_gaps = torch.zeros(iterations)
    relative_dual_pseudo_duality_gaps = torch.zeros(iterations)
    MMD = torch.zeros(iterations, device=my_device)  # mmd(X, Y)^2 during the algorithm
    W2 = torch.zeros(iterations, device=my_device)
    spread = torch.zeros(iterations, device=my_device)

    if annealing:
        assert B
        lambdas = np.zeros(iterations)  # regularization parameter during the algorithm (relevant for annealing) 
        lower_bds_lambd = np.zeros(iterations)
        print(f'embedding constant is {round(B,2)}')

    kyy = kern(Y[:, None, :], Y[None, :, :], s)
    kyy_norm = torch.norm(kyy)

    snapshots = 1e2*np.arange(1, 10)
    for n in tqdm(range(iterations)):
        spread[n] = (X-X.mean(dim=0)).norm(dim=1).mean() # 1st moment of X

        # plot the particles
        if plot and not n % 1000 or n in snapshots:
            if annealing:
                img_name = f'/Reg_{div.name}{a}flow,annealing,tau={step_size},{kernel},{s},{N},{M},{max_time},{target_name}-{n}.png'
            else:
                img_name = f'/Reg_{div.name}{a}flow,lambd={lambd},tau={step_size},{kernel},{s},{N},{M},{max_time},{target_name}-{n}.png'

            if d in [2, 3]:
                if n == 0 or not arrows:
                    h_star_grad = None
                plot_particles(X.cpu(), target, h_star_grad, target_name, img_name, folder_name, arrows)

        # construct kernel matrix
        kyx = kern(X[None, :, :], Y[:, None, :], s)
        kxx = kern(X[:, None, :], X[None, :, :], s)
        row_sum_kyx = kyx.sum(dim=1)  # tensor of shape (M, )
        sum_kxx = kxx.sum()
        kyx_sum = row_sum_kyx.sum()
        kyy_sum = kyy.sum()

        if div.reces(a) != float('Inf'):
            K_upper = torch.cat((kyy, kyx), dim=1)
            K_lower = torch.cat((kyx.T, kxx), dim=1)
            K = torch.cat((K_upper, K_lower))

        # calculate MMD(X, Y)^2 and W2 metric between particles and target
        mmd = kyy_sum / N ** 2 + sum_kxx / M ** 2 - 2 * kyx_sum / (N * M)
        MMD[n] = mmd
        if compute_W2:
            M2 = ot.dist(Y, X, metric='sqeuclidean')
            W2[n] = ot.emd2(a, b, M2)

        # annealing
        if annealing and div.reces(a) not in [0.0, float('inf')]:
            lower_bds_lambd[n] = 2 * torch.sqrt(2*MMD[n]) * B / div.reces(a)
            if not (lambd > lower_bds_lambd[n]):
                print("Condition is not fulfilled")
            if annealing_factor > 0 and n in [5e3, 1e4, 2e4]:
                lambd /= annealing_factor
                if verbose:
                    print(f"new lambda = {lambd}")
            elif annealing_factor == 0 and lambd > 1e-2:
                lambd = lower_bds_lambd[n] + 1e-4
            lambdas[n] = lambd

        # objectives for the case div.reces(a) = float('inf')
        def primal_objective(q):  # q.shape = (M, )
            convex_term = 1/M * torch.sum(div.fnc(q, a))
            quadratic_term = q @ (kyy @ q) - 2 * (M / N) * (q.t() @ row_sum_kyx)
            return convex_term + 1/(2 * lambd * M * M) * quadratic_term

        def primal_jacobian(q):
            convex_term = 1/M * div.der(q, a)
            linear_term = kyy @ q - M / N * row_sum_kyx
            return convex_term + 1/(lambd * M * M) * linear_term

        # primal objective for div.reces(a) < float('inf')
        def primal_objective_fin_rec(q):  # q.shape = (M+N, )
            convex_term = 1/M * torch.sum(div.fnc(q[:M], a))
            linear_term = div.reces(a) / M * torch.sum(q[M:])
            quadratic_term = 1/(2 * lambd * M * M) * q @ (K @ q)
            return convex_term + linear_term + quadratic_term

        def dual_objective(b):  # b.shape = (M, )
            term_1 = kyy @ b + 1 / (lambd * N) * row_sum_kyx
            term_2 = 1 / M * div.conj(term_1, a).sum()
            quadratic_term = lambd / 2 * b @ (kyy @ b)
            return term_2 + quadratic_term

        def dual_objective_fin_rec(b):  # b.shape = (M+N, )
            linear_term = 1 / N * (K_lower @ b).sum() 
            # maybe use Kyxsum and kxxsum instead?
            convex_term = 1 / M * (div.conj(K_upper @ b, a)).sum()
            quadratic_term = lambd / 2 * b @ (K @ b)
            return - linear_term + convex_term + quadratic_term

        def tight_dual_objective(b):
            term_1 = kyy @ b + 1 / (lambd * N) * row_sum_kyx
            dfnutilde = lambda lam: 1/M * div.conj(term_1 + lam, a).sum() - lam
            lam = torch.nn.Parameter(torch.tensor(0.0))
            # if not div.conj_der:
            #     print('f^* is not differentiable')
            # else:
            optimizer = torch.optim.LBFGS([lam], lr=1, max_iter=20)

            def closure():
                optimizer.zero_grad()  # Clear gradients
                loss = dfnutilde(lam)  # Compute the loss
                loss.backward(retain_graph=False)  # Compute gradients
                return loss
            for _ in range(10):  # Perform optimization
                loss = optimizer.step(closure)
            quadratic_term = lambd / 2 * b @ (kyy @ b)
            return loss + quadratic_term

        if primal:
            if tight:  # tight formulation
                # mirror descent on M * unit simplex in R^M
                # with differen line search methods for choosing step size
                if n == 0:
                    q = torch.ones(M, device=my_device)  # initial vector
                    norm_factor = torch.norm(primal_objective(q))
                q_end = q
                f_recs = torch.zeros(number_of_steps_mirror*100, device=my_device)

                rel_res = torch.norm(primal_jacobian(q)) / norm_factor
                iter_diff = torch.norm(q)
                k = 0
                # rel_res > M * 1e-5 or 
                while iter_diff > 1e-3 and k < number_of_steps_mirror*100 - 1:
                # for k in range(number_of_steps_mirror):
                    fx = primal_objective(q)
                    if line_search == 'Polyak':
                        sig, l = 0, 0
                        f_recs[k] = fx.item()
                        f_rec = torch.min(f_recs[:k+1])
                        idx = torch.zeros(2*number_of_steps_mirror+1, device=my_device)
                        idx[0] = 1
                    eta = np.log(k+2)/(k+2)  # initial guess
                    if k == 0: 
                        p = primal_jacobian(q)
                    if line_search == 'armijo':
                        t = - c*torch.dot(p, q)
                        eta = armijo_search(primal_objective, q, eta, p, fx, t, tau)
                    elif line_search == 'Polyak':
                        if primal_objective(q) <= f_recs[int(idx[l])] - 1/2*delta:
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
                    elif line_search == 'const':
                        eta = 1e-2
                    elif line_search == 'two_way':  # use two-way backtracking search
                        eta = two_way_backtracking_line_search(primal_objective, primal_jacobian, q, -p, alpha0=eta)    
                    q_prev = q
                    q = q * torch.exp(- eta * p)  # gradient step
                    q /= 1/M * torch.sum(q)  # projection onto the simplex
                    p = primal_jacobian(q)  # new gradient
                    rel_res = torch.norm(p) / norm_factor
                    iter_diff = torch.norm(q - q_prev)
                    q_end += q
                    k += 1
                if verbose:
                    print(f"Relative residual is {rel_res.item():.4f}, iteration difference is {iter_diff.item():.4f}")
                    print(f"Converged in {k + 1} iterations.")

                if averaging:
                    q_end /= number_of_steps_mirror
                    prim_value = primal_objective(q)
                else:
                    prim_value = primal_objective(q)

            else:  # use FISTA for non-tight formulation
                t = np.array(1.0, dtype=np.float64)
                if n == 0:
                    if div.reces(a) != float('inf'):
                        q = 1/10 * torch.ones(M + N, device=my_device)
                        objective = primal_objective_fin_rec
                    else:
                        q = 1/10 * torch.ones(M, device=my_device)
                        objective = primal_objective
                if div.reces(a) != float('inf'):
                    K_norm = torch.sqrt(kyy_norm**2 +
                                        2*torch.norm(kyx)**2
                                        + torch.norm(kxx)**2)
                else:
                    K_norm = kyy_norm
                for k in range(max_iter_FFBS):
                    q_old = q
                    t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t**2))
                    y = q + (t - 1.0)/t_next * (q - q_old)
                    if div.reces(a) != float('inf'):
                        y_tmp = y - K @ y / K_norm
                        q = torch.cat((
                            div.prox(y_tmp[:M+1], a, lambd * M / K_norm),
                            torch.maximum(torch.tensor(-M/N), y_tmp[M+1:] - div.reces(a) * lambd * M / K_norm)
                            ))
                    else:
                        y_tmp = y - 1 / K_norm * kyy @ y + M / (N * K_norm) * row_sum_kyx
                        eta = lambd * M / K_norm
                        q = div.prox(y_tmp, a, eta).to(dtype=torch.float64)
                    if torch.norm(y_tmp) < 1e-3 and torch.norm(q - q_old) / torch.norm(q) < 1e-3:
                        if verbose:
                            print(f'Converged in {k + 1} iterations')
                            print(f'Iter_diff = {torch.norm(q - q_old).item()}, residual = {torch.norm(y_tmp).item()}')
                        break
                    t = t_next
                prim_value = objective(q)

            primal_values[n] = prim_value + 1 / (2 * lambd * N**2) * sum_kxx
            if save_opts and not n % 1e5:
                torch.save(q, f'{folder_name}/q_at_{n}.pt')
        if div.reces(a) != float('inf'):
            dual_obj = dual_objective_fin_rec
            primal_obj = primal_objective_fin_rec
        else:
            if tight:
                dual_obj = tight_dual_objective
            else:
                dual_obj = dual_objective

        if dual:  # solve dual problem
            if tight:
                raise Exception('In the tight formulation, the dual ' +
                                'formulation is very costly to optimize')
            if n == 0:  # set initial values
                if div.reces(a) != float('inf'):
                    warm_start_b = - 1/(lambd * M) * torch.zeros(M + N, device=my_device)
                else:
                    warm_start_b = - 1/(lambd*N) * 1/100 * torch.ones(M, device=my_device)
            else:
                warm_start_b = b

            if not div.der:
                raise Exception('f is not differentiable')
            b = torch.nn.Parameter(warm_start_b)  # Initial guess for optimization variable
            optimizer = torch.optim.LBFGS([b], lr=1, max_iter=12)

            def closure():
                optimizer.zero_grad()  # Clear gradients
                loss = dual_obj(b)  # Compute the loss
                loss.backward()  # Compute gradients
                return loss
            for _ in range(8):  # Perform optimization
                loss = optimizer.step(closure)
            b = b.detach()
            dual_values[n] = - loss.item()
            if div.reces(a) == float('inf'):
                dual_values[n] += 1 / (2 * lambd * N**2) * sum_kxx

            if save_opts and not n % 1e5:
                torch.save(b, f'{folder_name}/b_at_{n}.pt')

        # calculate (pseudo) duality gaps
        if dual and not primal:
            pseudo_primal_solver = - lambd * M * b
            pseudo_dual_values[n] = primal_obj(pseudo_primal_solver) + 1 / (2 * lambd * N**2) * sum_kxx
        if primal:
            pseudo_dual_solver = - 1 / (lambd * M) * q
            pseudo_primal_values[n] = - dual_obj(pseudo_dual_solver) + 1 / (2 * lambd * N**2) * sum_kxx
        if primal and dual:  # calculate duality gaps
            duality_gaps[n] = torch.abs(dual_values[n] - primal_values[n])
            relative_duality_gaps[n] = duality_gaps[n] / (torch.min(primal_values[n], dual_values[n])+1e-10)
            if duality_gaps[n] > gap_tol and n > 10:
                print(f'Iter {n}: duality gap = {duality_gaps[n]:.3f} > {gap_tol} = tol.')
            if relative_duality_gaps[n] > relative_gap_tol and n > 10:
                print(f'Iter {n}: relative duality gap = {relative_duality_gaps[n]:.3f} > {relative_gap_tol} = tol.')

        if primal and not dual:  # calculate pseudo-duality gaps
            '''
            if tight:
                pseudo_dual_values[n] = - tight_dual_objective(- 1/(lambd * M) * q) + 1 / (2 * lambd * N**2) * sum_kxx
            else:
                if div.reces(a) != float('inf'):
                    pseudo_dual_values[n] = - dual_objective(1/(lambd * N) * torch.concatenate((q, - M / N * torch.ones(N, device=my_device))))
                else:
                    pseudo_dual_values[n] = - dual_objective(- 1/(lambd * M) * q)
            '''
            primal_pseudo_duality_gaps[n] = torch.abs(pseudo_primal_values[n] - primal_values[n]).item()
            relative_primal_pseudo_duality_gaps[n] = primal_pseudo_duality_gaps[n] / (torch.min(pseudo_primal_values[n], primal_values[n])+1e-10)
            if primal_pseudo_duality_gaps[n] > pseudo_gap_tol and n > 10:
                print(f'Iter {n}: primal pseudo duality gap = {primal_pseudo_duality_gaps[n]:.4f} > tolerance = {pseudo_gap_tol}.')
            if relative_primal_pseudo_duality_gaps[n] > relative_pseudo_gap_tol and n > 10:
                print(f'Iter {n}: relative primal pseudo duality gap = {relative_primal_pseudo_duality_gaps[n]:.4f} > tolerance = {relative_pseudo_gap_tol}.')
        if dual and not primal:
            '''
            if div.reces(a) != float('inf'):
                pseudo_dual_values[n] = primal_objective(1/(lambd * N) * torch.concatenate((q, - M / N * torch.ones(N, device=my_device))))
            else:
            '''
            dual_pseudo_duality_gaps[n] = torch.abs(pseudo_dual_values[n] - dual_values[n])
            relative_dual_pseudo_duality_gaps[n] = dual_pseudo_duality_gaps[n] / (torch.min(pseudo_dual_values[n], dual_values[n])+1e-10)

            if dual_pseudo_duality_gaps[n] > pseudo_gap_tol and n > 10:
                print(f'Iter {n}: dual pseudo duality gap = {dual_pseudo_duality_gaps[n]:.4f} > tolerance = {pseudo_gap_tol}.')
            if relative_dual_pseudo_duality_gaps[n] > relative_pseudo_gap_tol and n > 10:
                print(f'Iter {n}: relative dual pseudo duality gap = {relative_dual_pseudo_duality_gaps[n]:.4f} > tolerance = {relative_pseudo_gap_tol}.')


        # gradient update
        Z = torch.cat((Y, X)) # shape = (M + N, 2)
        if dual and not primal:
            q = pseudo_primal_solver
        # kern_der(X, Z, s).shape = (M + N, N, 2)
        assert torch.isfinite(q).all
        if div.reces(a) != float('inf'):
            temp = q.view(M+N, 1, 1) * kern_der(X, Z, s)
        else:
            qtilde = torch.cat((q, - M / N * torch.ones(N, device=my_device)))
            temp = qtilde.view(M+N, 1, 1) * kern_der(X, Z, s)  # shape = (M + N, N, 2)

        unscaled_h_star_grad = torch.sum(temp, dim=0)
        h_star_grad = - 1 / (lambd * M) * unscaled_h_star_grad
        Dfnulambda_grad = 1 / N * torch.norm(unscaled_h_star_grad)**2
        X -= step_size * h_star_grad

        if save_opts and not n % 1e5:
            torch.save(h_star_grad, f'{folder_name}/h_star_grad_at_{n}.pt')
        if not n % 1e4 or n in snapshots:
            torch.save(Y, f'{folder_name}/Y_at_{n}.pt')
        cond3 = Dfnulambda_grad < 1e-3
        if cond3:
            print(f'time derivative of the objective = {Dfnulambda_grad.item():.4f} < 1e-3. Stopping iteration')
            break
    # save results and plot time lines
    suffix = f'timeline,{lambd},{step_size},{N},{M},{kernel},{s},{max_time},{target_name}'
    torch.save(spread, folder_name + f'/Reg_{div.name}-{a}_moment_{suffix}.pt')
    torch.save(primal_values, folder_name + f'/Reg_{div.name}-{a}_primal_div_value_{suffix}.pt')
    torch.save(dual_values, folder_name + f'/Reg_{div.name}-{a}_dual_div_value_{suffix}.pt')
    torch.save(MMD, folder_name + f'/Reg_{div.name}-{a}_MMD_{suffix}.pt')
    if compute_W2:
        torch.save(W2, folder_name + f'/Reg_{div.name}-{a}_DivW2_{suffix}.pt')
    torch.save(duality_gaps, folder_name + f'/Reg_{div.name}-{a}_duality_gaps_{suffix}.pt')
    torch.save(relative_duality_gaps, folder_name + f'/Reg_{div.name}-{a}_rel_duality_gaps_{suffix}.pt')
    torch.save(relative_primal_pseudo_duality_gaps, folder_name + f'/Reg_{div.name}-{a}_rel_prim_duality_gaps_{suffix}.pt')
    torch.save(relative_dual_pseudo_duality_gaps, folder_name + f'/Reg_{div.name}-{a}_rel_dual_duality_gaps_{suffix}.pt')
    torch.save(primal_pseudo_duality_gaps, folder_name + f'/Reg_{div.name}-{a}_prim_pseudo_duality_gaps_{suffix}.pt')
    torch.save(dual_pseudo_duality_gaps, folder_name + f'/Reg_{div.name}-{a}_dual_pseudo_duality_gaps_{suffix}.pt')
    torch.save(relative_dual_pseudo_duality_gaps, folder_name + f'/Reg_{div.name}-{a}_rel_dual_pseudo_duality_gaps_{suffix}.pt')       
    if timeline:  # plot MMD, objective value, and W2 along the flow
        suffix = f'timeline,{a},{lambd},{step_size},{kernel},{s}'
        plot_MMD(MMD.cpu().numpy(), f'/{div.name}_MMD_{suffix}.png', folder_name)
        primals = primal_values.cpu().numpy() if primal else pseudo_primal_values[n].cpu().numpy()
        duals = dual_values.cpu().numpy() if dual else pseudo_dual_values.cpu().numpy()
        plot_func_values(primals, duals, lambd, f'/{div.name}_objective_{suffix}.png', folder_name)
        if annealing:
            plot_lambdas(lambdas, lower_bds_lambd, f'/{div.name}_lambdas_{suffix}.png', folder_name)
        if compute_W2:
          plot_W2(W2.cpu().numpy(), f'/{div.name}_W2_timeline_{suffix}.png')
        if primal and dual:
            plot_gaps(duality_gaps.cpu().numpy(), relative_duality_gaps.cpu().numpy(), f'/{div.name}_duality_gaps_{suffix}.png', folder_name)
        elif primal and not dual:
            plot_all_gaps(relative_primal_pseudo_duality_gaps.detach().cpu().numpy(), primal_pseudo_duality_gaps.cpu().numpy(), f'/{div.name}_duality_gaps_{suffix}.png', folder_name)
        elif dual and not primal:
            plot_all_gaps(relative_dual_pseudo_duality_gaps.cpu().numpy(), dual_pseudo_duality_gaps.cpu().numpy(), f'/{div.name}_duality_gaps_{suffix}.png', folder_name)
    end = time.time()
    print(f'The whole algorithm took {end - start:.2f} seconds')
    return primal_values.cpu().numpy(), MMD.cpu().numpy(), W2.cpu().numpy(), spread.cpu().numpy()


_, _, _, spread= MMD_reg_f_div_flow(tight=False)
# _, _, _, spread_tight= MMD_reg_f_div_flow(tight=True)
# fig, ax = plt.subplots()
# plt.plot(range(len(spread))[100:], spread[100:], label='non-tight')
# plt.plot(range(len(spread))[100:], spread_tight[100:], label='tight')
# plt.xlabel('iterations')
# plt.ylabel(r'$|\mathbb{E}[|X - \mathbb{E}[X] |]|$')
# plt.legend()
# plt.yscale('log')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8)))
# plt.savefig('spread_tight_vs_non-tight.png', dpi=300, bbox_inches='tight')