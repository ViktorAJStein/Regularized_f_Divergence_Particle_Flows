import torch
import numpy as np
import scipy as sp

def reLU(x):
    return 1/2*(x+np.abs(x))

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
