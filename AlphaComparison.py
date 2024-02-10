'''
Example file of how to plot comparisons of the particle flows with the Tsallis-alpha divergence for different values of alpha
'''
from MMD_reg_fDiv_ParticleFlows_CUDA import *

def AlphaComparison(
    sigma = .5,
    step_size = 1e-1,
    max_time = 1000000,
    lambd = 1e-0,
    N = 300*3,
    kern = IMQ,
    kern_der = IMQ_der,
    target_name = 'two_lines',
    alphas = [1, 3/2, 2, 5/2, 3, 4, 5, 15/2, 10],
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
            plot=True, timeline=True, gif=True, arrows=False, compute_W2 = compute_W2, compute_KALE = compute_KALE) #, st = k)
           
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
   
AlphaComparison()
