import os
os.environ['CUDA_PATH'] = '/work/stein/torch_Johannes/include/'
from warnings import warn
from PIL import Image
import torch
import ot
import numpy as np
import scipy as sp
import time
import matplotlib
matplotlib.rcParams["axes.formatter.useoffset"] = False
import matplotlib.pyplot as plt
from line_profiler import LineProfiler

torch.set_default_dtype(torch.float64) # set higher precision
use_cuda = torch.cuda.is_available() # shorthand
print(f'CUDA available = {use_cuda}')

def make_folder(name):
    try:
        os.mkdir(name)
        print(f"Folder '{name}' created successfully.")
    except FileExistsError:
        print(f"Folder '{name}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}.")

def rotate_point(point, angle):
    x, y = point
    angle_rad = angle * (np.pi / 180.0)  # degrees to radians
    new_x = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    new_y = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return (new_x, new_y)

def rotate_points(points, angle):
    return np.array([rotate_point(point, angle) for point in points])


def neals_funnel(n_samples, st=314):
    # Generate samples from Neal's funnel
    rs = np.random.RandomState(st) # RandomState has to be re-initialized every time in order for the next line to always yield the same result
    y = rs.normal(0, 2, size=n_samples)
    x = rs.normal(0, np.exp(y/3), size=n_samples)
    y = (y+7.5)
    return np.column_stack((x, y))

def generate_data(N, st = 42, r = .3, delta = .5):
    '''
    1. Generate three rings target, each ring has N points sampled uniformly.
    The rings have a radius of r and a separation of _delta.
    
    2. Generate prior, which is a Gaussian with very small variance centered at
    leftmost point of the rightmost ring.
    Returns
    -------
    X : np.array, shape = (3*N, 2)
        target.
    Y : np.array, shape = (3*N, 2)
        prior.

    '''
    # TODO: convert to pytorch code
    X = np.c_[r * np.cos(np.linspace(0, 2 * np.pi, N + 1)), r * np.sin(np.linspace(0, 2 * np.pi, N + 1))][:-1]  # noqa
    # X = torch.cat(
    #    ( (r * torch.cos(torch.linspace(0, 2 * np.pi, N + 1))).unsqueeze(1),
    #    (r * torch.sin(torch.linspace(0, 2 * np.pi, N + 1))).unsqueeze(1)), dim = 1)[:-1]  # noqa
    for i in [1, 2]:
        X = np.r_[X, X[:N, :]-i*np.array([0, (2 + delta) * r])]
        #X = torch.cat(
        #  (X.unsqueeze(0),
        #  (X[:N, :]-torch.tensor([0, (2 + delta) * r])).unsqueeze(0)),
        #dim = -1)[0]
    rs = np.random.RandomState(st) # RandomState has to be re-initialized every time in order for the next line to always yield the same result
    Y = rs.standard_normal((N*(2+1), 2)) / 100 - np.array([0, r])

    Y = torch.from_numpy(Y).to(torch.float64)
    X = torch.from_numpy(X).to(torch.float64)
    return X, Y

# def generate_points(N, n = 3, r = .3, delta = .25):
#     '''
#     Generate the target consisting of N points arranged equidistantly on n circles,
#     each with radius r and separation delta between each ring.
# 
#     Returns
#     -------
#     X : torch tensor of shape (N, 2)
#         point cloud described above.
# 
#     '''
# 
#     if N % n != 0:
#         raise ValueError("N must be divisible by n for even distribution.")
# 
#     points_per_ring = N // n
#     points = []
# 
#     for i in range(n):
#         theta_values = torch.linspace(0, 2 * math.pi, points_per_ring + 1)[:-1]
#         radius_values = torch.full((points_per_ring,), r)
# 
#         # Calculate x-coordinates with spacing delta between rings
#         x = radius_values * torch.cos(theta_values) - i * (2+delta)*r
#         y = radius_values * torch.sin(theta_values)
# 
#         # Append the points to the result list
#         points.append(torch.stack((x, y), dim=1))
# 
#     # Concatenate the points from all rings
#     result = torch.cat(points, dim=0)
#     result = result[:, [1, 0]] # switch columns
# 
#     return result


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




# Gaussian kernel with width s
def gauss(x, y, s):
    # if isinstance(x, torch.Tensor):
    return (-1 / (2*s) * (x - y) ** 2).sum(axis=-1).exp()
    # else:
    #     return np.exp((-1 / (2*s) * (x - y) ** 2).sum(axis=-1))

# derivative of Gaussian kernel    
def gauss_der(x, y, s):
    diff = y[:,None, :] - x[None,:, :]
    return 1 / s * (-1 / (2*s) * torch.linalg.vector_norm(diff, dim=2, keepdim=True)**2).exp() * diff


def IMQ(x, y, s):
    return (s + (((x - y) ** 2)).sum(axis=-1)) ** -(1/2)
    
def IMQ_der(x, y, s):
    diff = y[:,None, :] - x[None,:, :]
    pref = (torch.linalg.vector_norm(diff, dim=2, keepdim=True)**2 + s) ** -(3/2)
    return pref * diff
    
def Matern(x, y, sigma): # nu = 3/2
    r = ((x - y) ** 2).sum(axis=-1)
    return (1 + torch.sqrt(3*r) / sigma) * (- torch.sqrt(3*r) / sigma).exp()
    #2**(1 - nu) / sp.special.gamma(nu) * (np.sqrt(2 * nu * r) / sigma)**nu * sp.special.kv(nu, np.sqrt(2 * nu * r)/ sigma)
    
def Matern_der(x, y, sigma): # nu = 3/2
    diff = y[:,None, :] - x[None,:, :]
    r = torch.linalg.vector_norm(diff, dim=2, keepdim=True)
    return 3/sigma**2 * (- 3 / sigma * r).exp() * diff
    
def Matern2(x, y, sigma): # nu = 5/2
    r = ((x - y) ** 2).sum(axis=-1)
    return (1 + torch.sqrt(5*r) / sigma + 5*r/(3*sigma**2) ) * (- torch.sqrt(5*r) / sigma).exp()

def Matern2_der(x, y, sigma): # nu = 5/2
    diff = y[:,None, :] - x[None,:, :]
    r = torch.linalg.vector_norm(diff, dim=2, keepdim=True)**2
    return 5/6 * 1/sigma**3 * (torch.sqrt(5*r) + sigma) * (- torch.sqrt(5*r) / sigma).exp() * diff
    
def compact(x, y, q): # this expression depends on the dimension of the data points being d = 2 (or more generally, that floor(d/2) = 1)
    r = torch.sqrt(((x - y) ** 2).sum(axis=-1))
    return torch.nn.functional.relu(1 - r)**(q + 2)
    
def compact_der(x, y, q):
    diff = y[:,None, :] - x[None,:, :]
    r = torch.linalg.vector_norm(diff, dim=2, keepdim=True)
    return diff/r*(q+2)/2*torch.nn.functional.relu(1 - r)**(q + 1)

def compact2(x, y, q):  # this expression depends on the dimension of the data points being d = 2
    r = torch.sqrt(((x - y) ** 2).sum(axis=-1))
    return torch.nn.functional.relu(1 - r)**(q + 3) * ( (q + 3)*r + 1 ) 
    
def compact2_der(x, y, q):
    diff = y[:,None, :] - x[None,:, :]
    r = torch.linalg.vector_norm(diff, dim=2, keepdim=True)
    return 1/2*diff*(q+3)*(q+4)*torch.nn.functional.relu(1 - r)**(q + 2)
    
def thin_plate_spline(x, y, sigma):
    tol=1e-16
    r = ((x - y) ** 2).sum(axis=-1)**(1/2)
    return r * torch.log(r**r + tol)

def thin_plate_spline_der(x, y, sigma):
    tol=1e-16
    diff = x[:,None, :] - y[None,:, :]
    r = torch.linalg.vector_norm(diff, dim=2, keepdim=True)
    return 1/2*diff*(torch.log(r**2 + tol) + 1)
    
def squared_dot(x, y, sigma):
    return 1/2*torch.dot(x,y)**2

def squared_dot(x, y, sigma):
    return y

def reLU(x):
    return 1/2*(x + np.abs(x))

# f-divergence generators and their derivatives
def tsallis_generator(x, alpha):
    return np.choose(x >= 0, [np.inf, ((x+1e-16)**alpha - alpha*x + alpha - 1)/(alpha - 1)])

def tsallis_generator_der(x, alpha):
    return np.choose(x >= 0, [np.inf, alpha / (alpha - 1) * ( (x+1e-16)**(alpha - 1) - 1)])
    
def kl_generator_hess(x):
    return np.choose(x > 0, [np.inf, 1/x])

def tsallis_generator_hess(x, alpha):
    return np.choose(x >= 0, [np.inf, alpha * (x+1e-16)**(alpha - 2)])
    
def tsallis_hess(x, alpha):
    if alpha != 1:
        return tsallis_generator_hess(x, alpha)
    else:
        return kl_generator_hess(x)

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

def kl_generator(x):
    return np.choose(x >= 0, [np.inf,np.choose(x > 0, [1, x*np.log(x+1e-16)-x+1])])

def kl_generator_der(x):
    return np.choose(x > 0, [np.inf, np.log(x+1e-16)])

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
        
def jeffreys(x, alpha):
    tol = 1e-16
    return np.choose(x > 0, [np.inf, (x - 1) * np.log(tol + x)])

def jeffreys_der(x, alpha):
    tol = 1e-16
    return np.choose(x > 0, [np.inf, (x - 1)/x + np.log(tol + x) ])

def jeffreys_conj(x, alpha):
    lambert = np.real(sp.special.lambertw(np.e**(1 - x)))
    return x - 2 + lambert + 1/lambert

def jeffreys_conj_der(x, alpha):
    return 1 / np.real(sp.special.lambertw(np.e**(1 - x)))

def chi(x, alpha):
    return np.choose(x >= 0, [np.inf, np.abs(x - 1) ** alpha])

def chi_der(x, alpha):
    return np.choose(x >= 0, [np.inf, alpha * np.abs(x - 1)**(alpha - 1) * np.sign(x - 1)])

def chi_conj(x, alpha):
    return np.choose(x >= - alpha, [-1, x + (alpha - 1) * (np.abs(x) / alpha)**(alpha/(alpha - 1))])
    
def chi_conj_der(x, alpha):
    return 0 # TODO

## divergences with non-finite conjugates
def reverse_kl(x, alpha):
    tol = 1e-30
    return np.choose(x > 0, [np.inf, x - 1 - np.log(x + tol)])

def reverse_kl_der(x, alpha):
    return np.choose(x > 0, [np.inf, (x-1)/x])

def jensen_shannon(x, alpha):
    tol = 1e-30
    return np.choose(x > 0, [np.inf, np.log(tol + x) - (x + 1) * np.log((x+1)/2) ])

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
    return np.where(y <= 1, np.maximum(y, -1), np.inf)

def tv_conj_der(x, alpha):
    return np.select([np.abs(x) <= 1], [1], default=0)


def MMD_reg_f_div_flow(
        alpha = 5,
        sigma = .05,
        N = 1002,
        lambd = .01,
        step_size = .001,
        max_time = 1,
        plot = True, # plot particles along the evolution
        arrows = False, # plots arrows at particles to show their gradients
        gif = True, # produces gif showing the evolution of the particles
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
        st = 42
        ):
    
    '''
    @return:    func_value:    list of length N, records objective value during simulation
                # elapsed_time:  elapsed time during the simulation in seconds
    '''
    

    iterations = int(max_time / step_size) + 1, # max number of iterations


    if div != tsallis and div != chi:
        alpha = ''
    
    kernel = kern.__name__
    divergence = div.__name__
    folder_name = f"{divergence},lambda={lambd},tau={step_size},{kernel},{sigma},{N},{max_time},{target_name}/{divergence},alpha={alpha},lambd={lambd},tau={step_size},{kernel},{sigma},{N},{mode},{max_time},{target_name}"
    make_folder(folder_name)
    
    p_start = time.time()    
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
        # target = generate_points(N) 
        # r = .3
        target, prior = generate_data(int(N/3), st=st)
        # mean and variance of prior distribution
        # m_p = torch.tensor([0, -.3])
        # v_p = 1 / 100 * torch.eye(d)
        # Draw samples from the prior distribution
        # multivariate_normal = torch.distributions.MultivariateNormal(m_p, v_p) # Create a MultivariateNormal distribution
        # prior = multivariate_normal.sample((N,)) # Generate samples
        # prior = rs.randn(N*(2+1), 2) / 100 - np.array([0, r])
        

    if target_name == 'two_lines': # two lines target
        u = int(N/2)
        
        torch.manual_seed(st) # fix randomness
        
        # layer 1
        vert = torch.rand(u)
        hori = torch.rand(u)
        l = torch.linspace(-1, 1, u) + vert
        squared  = l**2 + hori
    
        # layer 2
        vert2 = torch.rand(u)
        hori2 = torch.rand(u)
        l2 = torch.linspace(-1.5, 1.5, u) + vert2
        squared2  = 1/2*(l2-1)**2 + hori2 - 4
    
        l = torch.cat((l, l2))
        squared = torch.cat((squared, squared2))
        target = torch.stack((l, squared)).transpose(0, 1)
        
        # mean and variance of prior distribution
        m_p = torch.tensor([0, 4.0])
        v_p = 1 / 2000 * torch.eye(d)


        # Draw samples from the prior distribution
        multivariate_normal = torch.distributions.MultivariateNormal(m_p, v_p) # Create a MultivariateNormal distribution
        prior = multivariate_normal.sample((N,)) # Generate samples
     
    p_end = time.time()
    if verbose: print(f"Generating prior and target took {p_end - p_start} seconds")
        
    if use_cuda:
      Y = prior.clone().to("cuda") # samples of prior distribution
      X = target.to("cuda") # samples of target measure 
    else:
      Y = prior.clone()
      X = target
      
    # prior.requires_grad = True

    #### now start particle descent
    iterations = int(iterations[0]) # reset iterations to be the int from the beginning
    # Keeping track of different values along the iterations
    func_values = [] # objective value during the algorithm
    dual_values = []
    pseudo_dual_values = []
    MMD = torch.zeros(iterations) # mmd(X, Y) during the algorithm
    # W1 = torch.zeros(iterations)
    W2 = torch.zeros(iterations)
    duality_gaps = []
    pseudo_duality_gaps = []
    # stable_relative_duality_gaps = []
    # stable_relative_pseudo_duality_gaps = []
    relative_duality_gaps = []
    relative_pseudo_duality_gaps = []
    dual_values = []    
    start_time = time.time()
    
    kxx = kern(X[:, None, :], X[None, :, :], sigma)
    a,b = torch.ones(N) / N, torch.ones(N) / N

    for n in range(iterations):
        # plot the particles ten times per unit time interval
        time1 = round(n*step_size, 1)
        if plot and not n % 1000 or n in 100*np.arange(1, 10):
            plot_start = time.time()
            plt.figure() 
            plt.plot(X.cpu()[:, 1], X.cpu()[:, 0], '.', color='orange', markersize = 2) # plot target
            plt.plot(Y.cpu()[:, 1], Y.cpu()[:, 0], '.', color='blue', markersize = 2) # plot particles
            if arrows:
                for i in range(len(Y)) and i > 0:
                    point = Y_CPU[i]
                    vector = - h_star_grad.cpu()[i]
                    plt.arrow(point[0], point[1], vector[0], vector[1], head_width=0.05, head_length=0.1, fc='k', ec='k', linewidth=.5)                        
            
            
            # if target_name == 'circles':
            #    plt.ylim([-.5, .5])
            #    plt.xlim([-2.0, .5])

            plt.gca().set_aspect('equal')
            plt.axis('off')
            
            time_stamp = int(time1*10)
            img_name = f'/MMD-Reg_{divergence}{alpha}_div_flow,lambd={lambd},tau={step_size},{kernel},{sigma},{N},{max_time},{target_name}-{n}.png'
            plt.savefig(folder_name + img_name, dpi=300, bbox_inches='tight')
            plt.close()
            plot_end = time.time()
            if verbose: print(f"Plotting took {plot_end - plot_start} seconds")
    
    
        if verbose: print(f"---------------------- Iteration {n} -----------------------")
        ### construct kernel matrix
        metric_st = time.time()
        
        kxy = kern(X[:, None, :], Y[None, :, :], sigma)
        kyy = kern(Y[:, None, :], Y[None, :, :], sigma)
        upper_row = torch.cat((kxx, kxy), dim=1)
        lower_row = torch.cat((kxy.t(), kyy), dim=1)
        K = torch.cat((upper_row, lower_row), dim=0).cpu().numpy()
        
        ## calculate MMD(X, Y), W1 and W2 metric between particles and target
        MMD[n] = 1/(2 * N**2) * (kxx.sum() + kyy.sum() - 2 * kxy.sum())
        # M1 = ot.dist(X, Y, metric='euclidean')
        M2 = ot.dist(X, Y, metric='sqeuclidean')
        # W1[n] = ot.emd2(a, b, M1)
        W2[n] = ot.emd2(a, b, M2)
        metric_end = time.time()
        if verbose:
            print(f'Calculating kernel matrix, MMD, W1 and W2 took {metric_end - metric_st} seconds')
        


        # primal objective is an N-dimensional function
        def primal_objective(q):
            convex_term = np.sum(div(q, alpha))
            tilde_q = np.concatenate((q, - np.ones(N)))
            quadratic_term = tilde_q.T @ K @ tilde_q
            return 1/N * convex_term + 1/(2 * lambd * N * N) * quadratic_term
            
        # def primal_objective_reduced(q):
        #     convex_term = np.sum(div(q, alpha))
        #     quadratic_term = q.T @ kxx @ q - 2 * (kxy @ q).sum()
    
        # jacobian of the above ojective function
        def primal_jacobian(q):
            convex_term = div_der(q, alpha)
            tilde_q = np.concatenate((q, - np.ones(N)))
            linear_term = upper_row.cpu().numpy() @ tilde_q
            return 1/N * convex_term + 1/(lambd * N * N) * linear_term
            
        # def prim_hess(q):
        #     convex_term = div_hess(q, alpha = alpha)
        #     return 1/N * convex_term + 1/(lambd * N * N) * kxx


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
         
        # def dual_hess(b):
        #     return
        
        if mode == 'primal' or mode == 'dual':
            if verbose: sp_start = time.time()

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
                bounds=[(1e-15, None) for _ in range(len(warm_start_q))],
                **opt_kwargs,
            )
            func_values.append(prim_value)
            # primal = sp.optimize.minimize(primal_objective, warm_start_q, method='L-BFGS-B', jac=primal_jacobian, options={'gtol': 1e-13}) 
            # q_np, func_value[n] = primal.x, primal.fun # solution vector of optimization problem
            if mode == 'dual' and alpha == '':
                b_np, minus_dual_value, _ = sp.optimize.fmin_l_bfgs_b(dual_objective, warm_start_b, fprime = dual_jacobian, **opt_kwargs) 
                dual_values.append(-minus_dual_value)
                if plot and not n % 10000:
                  torch.save(torch.from_numpy(b_np), f'{folder_name}/b_at_{n}.pt')
            sp_end = time.time()
            if verbose: print(f"scipy took {sp_end - sp_start} seconds")
            start_ne = time.time()

            
            pseudo_dual_value = - dual_objective(1/(lambd * N) * np.concatenate((-q_np, np.ones(N))))
            pseudo_dual_values.append(pseudo_dual_value)
            pseudo_duality_gap = np.abs(prim_value - pseudo_dual_value)
            pseudo_duality_gaps.append(pseudo_duality_gap)
            relative_pseudo_duality_gap = pseudo_duality_gap / np.min((np.abs(prim_value), np.abs(pseudo_dual_value)))
            relative_pseudo_duality_gaps.append(relative_pseudo_duality_gap)
            # stable_relative_pseudo_duality_gap = pseudo_duality_gap / np.max((1e-16, np.min((np.abs(prim_value), np.abs(pseudo_dual_value))) )) # 1e-16 is for stability
            # stable_relative_pseudo_duality_gaps.append(stable_relative_pseudo_duality_gap)
            pseudo_gap_tol = 1e-2
            relative_pseudo_gap_tol = 1e-2
            # stable_relative_pseudo_gap_tol = 1e-2
            if pseudo_duality_gap > pseudo_gap_tol and verbose:
                  warn(f'Iteration {n}: pseudo-duality gap {pseudo_duality_gap} is larger than tolerance {pseudo_gap_tol}.')
            if relative_pseudo_duality_gap > relative_pseudo_gap_tol and verbose:
                  warn(f'Iteration {n}: relative pseudo-duality gap {relative_pseudo_duality_gap} is larger than tolerance {relative_pseudo_gap_tol}.')
            # if stable_relative_pseudo_duality_gap > stable_relative_pseudo_gap_tol and verbose:
            #       warn(f'Iteration {n}: stable relative pseudo-duality gap {stable_relative_pseudo_duality_gap} is larger than tolerance {stable_relative_pseudo_gap_tol}.')
                  
            if mode == 'dual' and not alpha == '':
              #b_np = dual.x
              dual_value = - minus_dual_value
              duality_gap = np.abs(prim_value - dual_value)
              duality_gaps.append(duality_gap)
              relative_duality_gap = duality_gap / np.min((np.abs(prim_value), np.abs(dual_value)))
              relative_duality_gaps.append(relative_duality_gap)
              # stable_relative_duality_gap = duality_gap / np.max(( 1e-9, np.min((np.abs(prim_value), np.abs(dual_value))) ))
              # stable_relative_duality_gaps.append(stable_relative_duality_gap)
              gap_tol = 1e-2
              relative_gap_tol = 1e-2
              # stable_relative_gap_tol = 1e-2    
              if relative_duality_gap > relative_gap_tol and verbose:
                  warn(f'Iteration {n}: relative duality gap {relative_duality_gap} is larger than tolerance {relative_gap_tol}.') 
              # if stable_relative_duality_gap > stable_relative_gap_tol and verbose:
              #     warn(f'Iteration {n}: stable relative duality gap {stable_relative_duality_gap} is larger than tolerance {stable_relative_gap_tol}.')     

            if use_cuda:  
              q_torch = torch.tensor(q_np, dtype=torch.float64, device="cuda") # torch version of solution vector
              
            else:
              q_torch = torch.tensor(q_np, dtype=torch.float64, device="cpu") 

            # save solution vector in every 100-th iteration (to conserve memory)
              if plot and not n % 10000:
                  torch.save(q_torch, f'{folder_name}/q_at_{n}.pt')
                  # plt.plot(q_torch.cpu())
                  # plt.savefig(f'{folder_name}/q_at_{n}.png', dpi=300, bbox_inches='tight')
                  # plt.close()
                  
            # gradient update
            g_start = time.time()
            temp = kern_der(Y, Y, sigma) - q_torch.view(N, 1, 1) * kern_der(Y, X, sigma)
            h_star_grad = 1 / (lambd * N) * torch.sum(temp, dim=0)            
            if plot and not n % 10000:
              torch.save(h_star_grad, f'{folder_name}/h_star_grad_at_{n}.pt')
              # plt.scatter(h_star_grad.cpu().numpy()[:, 0], h_star_grad.cpu().numpy()[:, 1])
              # plt.savefig(f'{folder_name}/h_star_grad_at_{n}.png', dpi=300, bbox_inches='tight')
              # plt.close()
            Y -= step_size * (1 + lambd) * h_star_grad
            g_end = time.time()
            if verbose: print(f"Gradient update took {g_end - g_start} seconds")
            # save position of particles in every 100-th iteration (to conserve memory)
            if not n % 1000 or n in 100*np.arange(1, 10):
                torch.save(Y, f'{folder_name}/Y_at_{n}.pt')
        
    end_time = time.time()
    elapsed_time = end_time - start_time
     
    if gif:
        output_name = f'/{divergence}{alpha}__flow,lambd={lambd},tau={step_size},{kernel},{sigma},{N},{max_time},{target_name}.gif'    
        create_gif(folder_name, output_name)
    
    
    torch.save(func_values, folder_name + f'/Reg_{divergence}-{alpha}_Div_value_timeline,{lambd},{step_size},{N},{kernel},{sigma},{max_time},{target_name}.pt')
    torch.save(MMD, folder_name + f'/Reg_{divergence}-{alpha}_Div_MMD_timeline,{lambd},{step_size},{N},{kernel},{sigma},{max_time},{target_name}.pt')
    torch.save(W2, folder_name + f'/Reg_{divergence}-{alpha}_DivW2_timeline,{lambd},{step_size},{N},{kernel},{sigma},{max_time},{target_name}.pt')
    if mode == 'dual':
        torch.save(duality_gaps, folder_name + f'/Reg_{divergence}-{alpha}_Divergence_duality_gaps_timeline,{lambd},{step_size},{N},{kernel},{sigma},{max_time},{target_name}.pt')
        torch.save(relative_duality_gaps, folder_name + f'/Reg_{divergence}-{alpha}_Divergence_rel_duality_gaps_timeline,{lambd},{step_size},{N},{kernel},{sigma},{max_time},{target_name}.pt')
    torch.save(pseudo_duality_gaps, folder_name + f'/Reg_{divergence}-{alpha}_Divergence_pseudo_duality_gaps_timeline,{lambd},{step_size},{N},{kernel},{sigma},{max_time},{target_name}.pt')
    torch.save(relative_pseudo_duality_gaps, folder_name + f'/Reg_{divergence}-{alpha}_Divergence_rel_pseudo_duality_gaps_timeline,{lambd},{step_size},{N},{kernel},{sigma},{max_time},{target_name}.pt')       
    if timeline:
        # plot MMD, objective value, and W2 along the flow
        fig, ax = plt.subplots()
        
        plt.plot(MMD.cpu().numpy())
        plt.xlabel('iterations')
        plt.ylabel(r'$d_{K}(\mu, \nu)$')
        plt.yscale('log')
        plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8)))
        plt.savefig(folder_name + f'/{divergence}_MMD_timeline,{alpha},{lambd},{step_size},{kernel},{sigma}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
        
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
        if not alpha == '': # and alpha > 1:
          plt.plot(duality_gaps, label='duality gap')
          plt.plot(relative_duality_gaps, '-.', label='relative duality gap')
          # plt.plot(stable_relative_duality_gaps, ':', label='stable relative duality gap')
        plt.plot(pseudo_duality_gaps, ':', label='pseudo duality gap')        
        # plt.plot(stable_relative_pseudo_duality_gaps, '-.', label='stable relative pseudo duality gap')
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
    return func_values, MMD, W2


def this_main(
    sigma = .05,
    step_size = 1e-3,
    max_time = 100,
    lambd = 1e-2,
    N = 300*3,
    kern = IMQ,
    kern_der = IMQ_der,
    target_name = 'two_lines',
    alphas = [1.01, 3/2, 2, 5/2, 3, 4, 5, 15/2, 10, 50],
    div = chi,
    div_der = chi_der,
    div_conj = chi_conj,
    div_conj_der = chi_conj_der,
    ):
    if div != tsallis and div != chi:
        alpha = ''
    kernel = kern.__name__
    diverg = div.__name__
    iterations = int(max_time / step_size)
    L = len(alphas)
    func_values = torch.zeros(L, iterations + 1)
    MMD_values = torch.zeros(L, iterations + 1)
    W2_values = torch.zeros(L, iterations + 1)
     
    folder = f'{diverg},lambda={lambd},tau={step_size},{kernel},{sigma},{N},{max_time},{target_name}'
    make_folder(folder)
    
    for k in range(L):
      func_values[k, :], MMD_values[k, :], W2_values[k, :] = MMD_reg_f_div_flow(
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
            plot=True, timeline=True, gif=False) #, st = k)
           
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

    
    # plot W2
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
   
this_main()