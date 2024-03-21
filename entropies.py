import torch
import numpy as np
import scipy as sp

def reLU(x):
    return 1/2*(x+np.abs(x))


def tsallis_generator(x, alpha):
    return np.choose(x >= 0, [np.inf, ((x+1e-30)**alpha - alpha*x + alpha - 1)/(alpha - 1)])

def tsallis_generator_der(x, alpha):
    return np.choose(x >= 0, [np.inf, alpha / (alpha - 1) * ( (x+1e-30)**(alpha - 1) - 1)])


# the conjugate f_alpha* of the entropy function f_alpha for alpha > 1
def tsallis_conj(x, alpha):
    if alpha != 1:
        return reLU((alpha - 1)/alpha * x + 1)**(alpha / (alpha - 1)) - 1
    else:
        return np.exp(x) - 1
   
def tsallis_conj_der(x, alpha):
    if alpha != 1:
        return reLU( (alpha - 1)/alpha * x + 1)**(1/(alpha - 1))
    else:
        return np.exp(x)

def kl_generator(x):
    return sp.special.xlogy(x, x) - x + 1

def kl_generator_der(x):
    return np.log(x)


def tsallis(x, alpha):
    if alpha != 1:
        return tsallis_generator(x,alpha)
    else:
       return kl_generator(x)
   
def tsallis_der(x, alpha):
    if alpha != 1:
        return tsallis_generator_der(x, alpha)
    else:
        return kl_generator_der(x)
        
def jeffreys(x, alpha):
    return sp.special.xlogy(x - 1, x)

def jeffreys_der(x, alpha):
    return (x - 1)/x + np.log(x)
    
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
    return np.choose(x >= - alpha, [0, 1 + (alpha)**(1/(1-alpha)) * np.abs(x)**(1/(alpha - 1)) * np.sgn(x) ])

# divergences with non-finite conjugates    
def lindsay(x, alpha):
    return (x - 1)**2 / (alpha + (1 - alpha)*x)
    
def lindsay(x, alpha):
    return ((1 - x) * (-alpha + (alpha - 1) * x - 1)) / ((alpha - 1) * x - alpha)^2
    
def perimeter(x, alpha):
    if alpha == 1:
        return jensen_shannon(x, alpha)
    elif alpha == 0:
        return 1/2*tv(x, alpha)
    else:
        return np.choose(x >= 0, [np.inf, np.sgn(alpha) / (1 - alpha) * ( (x**(1/alpha) + 1)**alpha - 2**(alpha - 1) * (x + 1) )])
        
def perimeter_der(x, alpha):
    if alpha == 1:
        return jensen_shannon_der(x, alpha)
    elif alpha == 0:
        return 1/2*tv_der(x, alpha)
    else:
        return np.choose(x > 0, [ np.sgn(alpha) / (1 - alpha) * ( (x**(-1/alpha) + 1)**(alpha - 1) - 2**(alpha - 1) )])
        

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

def power(x, alpha):
    if alpha != 0:
        return 1/alpha * tsallis(x)
    else:
        return reverse_kl(x)
        
def power_der(x, alpha):
    if alpha != 0:
        return 1/alpha * tsallis_der(x)
    else:
        return reverse_kl_der(x)
    
def tv(x, alpha):
    return np.choose(x >= 0, [np.inf, np.abs(x - 1)])

def tv_der(x, alpha):
    return np.choose(x >= 0, [np.inf, np.sign(x - 1)])
    
def tv_conj(y, alpha):
    return np.where(y <= 1, np.maximum(y, -1), np.inf)

def tv_conj_der(x, alpha):
    return np.select([np.abs(x) <= 1], [1], default=0)
    
def matusita(x, alpha):
    return np.abs(1 - x**(alpha))**(1/alpha)
    
def matusita_der(x, alpha):
    return x**(alpha-1) * (x**alhpa - 1) * np.abs(1 - x**alpha)**(1/alpha - 2)
    
def marton(x, alpha):
    return np.max(0, 1 - x)**2
    
def marton_der(x, alpha):
    return 2*np.max(0, 1 - x)
        
# define recession constants
def rec_const(div, alpha = None):
    if div == 'power':
        if alpha >= 1:
            return float('inf')
        elif alpha < 1 and alpha != 0:
            return 1 / (1 - alpha)
        elif alpha == 0:
            return 1

    if div == 'tsallis':
        if alpha >= 1:
            return float('inf')
        elif 0 < alpha < 1:
            return alpha / (1 - alpha)

    if div in ['jeffreys', 'chi']:
        return float('inf')
     
    if div == 'lindsay':
        return 1/(1 - alpha)
        
            
    if div in ['tv', 'reverse_KL', 'matusita', 'kafka']:
        return 1
    
    if div == 'marton':
        return 0
        
    if div == 'jensen_shannon':
        return np.log(2)
        
    if div == 'perimeter':
        if alpha > 0:
            return 1/(1 - alpha) * (1 - 2**(alpha - 1))
        elif alpha < 0:
            return 1/(1 - alpha) * 2**(alpha - 1)
        elif alpha == 0:
            return 1/2
        elif alpha == 1:
            return np.log(2)