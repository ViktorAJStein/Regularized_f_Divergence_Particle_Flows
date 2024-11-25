import torch
import numpy as np
import scipy as sp


def lambertw_ext(x):
    mask = x > -1 / np.exp(1)
    result = np.empty_like(x)
    result[mask] = sp.special.lambertw(x[mask])
    result[~mask] = sp.special.lambertw(x[~mask], k=-1)
    return result

def reLU(x):
    return 1/2*(x+np.abs(x))

    
    
class entr_func:
    def __init__(self, fnc, der, hess, conj, conj_der, prox, reces):
        self.name = fnc.__name__
        self.fnc = fnc
        self.der = der
        self.hess = hess
        self.conj = conj
        self.conj_der = conj_der
        self.prox = prox
        self.reces = reces
        

def prox_newton(entr, eta, x, threshold=1e-7):
    y = 100
    while torch.abs(entr.der(y) + 1/eta * (y - x)) > threshold:
        y -= (eta*entr.der(y) + y - x)/(eta*entr.hess(y) + 1)


def tsallis_generator(x, a):
    return np.choose(x >= 0, [np.inf, ((x+1e-30)**a - a*x + a - 1)/(a - 1)])

def tsallis_generator_der(x, a):
    return np.choose(x >= 0, [np.inf, a / (a - 1) * ( (x+1e-30)**(a - 1) - 1)])


# the conjugate f_a* of the entropy function f_a for a > 1
def tsallis_conj(x, a):
    if a != 1:
        return reLU((a - 1)/a * x + 1)**(a / (a - 1)) - 1
    else:
        return np.exp(x) - 1
   
def tsallis_conj_der(x, a):
    if a != 1:
        return reLU( (a - 1)/a * x + 1)**(1/(a - 1))
    else:
        return np.exp(x)

def kl_generator(x):
    if isinstance(x, torch.Tensor):
        return torch.xlogy(x, x) - x + 1
    elif isinstance(x, np.ndarray):
        return sp.special.xlogy(x, x) - x + 1

def kl_generator_der(x):
    return np.log(x+1e-30)


def tsallis(x, a):
    if a != 1:
        return tsallis_generator(x,a)
    else:
       return kl_generator(x)
   
def tsallis_der(x, a):
    if a != 1:
        return tsallis_generator_der(x, a)
    else:
        return kl_generator_der(x)
        
def tsallis_hess(x, a):
    if a != 1:
        return 1/x
    else:
        return a * x**(a - 2)

def tsallis_torch(x, a):
    if a != 1:
        return torch.where(x >= 0, ((x+1e-30)**a - a*x + a - 1)/(a - 1), float("Inf"))
    else:
       return kl_generator(x)

def tsallis_der_torch(x, a):
    if a != 1:
        return torch.where(x >= 0, a / (a - 1) * ( (x+1e-30)**(a - 1) - 1), float("Inf"))
    else:
        return torch.log(x)
        
def tsallis_prox(x, a, eta):
    eta = eta.cpu().numpy()
    x = x.cpu().numpy()
    if a != 1:
        pass  # TODO!
    else:
        return eta * lambertw_ext(1/eta * np.exp(x/eta))

def tsallis_rec(a):
    if a >= 1:
        return float('inf')
    elif 0 < a < 1:
        a / (1 - a)
        
tsallis = entr_func(tsallis, tsallis_der, tsallis_hess, tsallis_conj, tsallis_conj_der, tsallis_prox, tsallis_rec)
                
def jeffreys(x, a):
    return sp.special.xlogy(x - 1, x)

def jeffreys_der(x, a):
    return (x - 1)/x + np.log(x)
    
def jeffreys_hess(x, a):
    return (x - 2) / (x - 1)**2
    
def jeffreys_conj(x, a):
    lambert = np.real(sp.special.lambertw(np.e**(1 - x)))
    return x - 2 + lambert + 1/lambert

def jeffreys_conj_der(x, a):
    return 1 / np.real(sp.special.lambertw(np.e**(1 - x)))

def jeffreys_prox(x, a, eta):
    f = lambda y : y - 1 + torch.xlogy(y, y) + 1 / eta * (y^2 - x * y)
    return newton(f)
    
jeffreys = entr_func(jeffreys, jeffreys_der, jeffreys_hess, jeffreys_conj, jeffreys_conj_der, jeffreys_prox, float('inf'))

    
def chi(x, a):
    return np.choose(x >= 0, [np.inf, np.abs(x - 1) ** a])

def chi_der(x, a):
    return np.choose(x >= 0, [np.inf, a * np.abs(x - 1)**(a - 1) * np.sign(x - 1)])

def chi_conj(x, a):
    return np.choose(x >= - a, [-1, x + (a - 1) * (np.abs(x) / a)**(a/(a - 1))])
    
def chi_conj_der(x, a):
    return np.choose(x >= - a, [0, 1 + (a)**(1/(1-a)) * np.abs(x)**(1/(a - 1)) * np.sign(x) ])

# divergences with non-finite conjugates    
def lindsay(x, a):
    assert a >= 0 and a <= 1
    return (x - 1)**2 / (a + (1 - a)*x)
    
def lindsay_der(x, a):
    return ((1 - x) * (-a + (a - 1) * x - 1)) / ((a - 1) * x - a)**2
    
def lindsay_hess(x, a):
    return 2 / (a * (1 - x) + x)**3
    
def lindsay_conj(x, a):
    return np.choose(x <= 1/(1 - a), [np.inf, ( a*(a - 1)*x - 2*np.sqrt( (a - 1)*x+1 ) + 2 )/(a - 1)**2 ])
    
def lindsay_conj_der(x, a):
    return np.choose(x < 1/(1 - a), [ np.inf, 1 / (a - 1) * (a - 1 / np.sqrt( (a - 1) * x + 1 ))])
    
def lindsay_prox(x, a, eta):
    F = lambda x:  ((1 - x) * (x * (a - 1) - a - 1))/ ((x + a - x * a)**2) + 1/eta * (x - y)
    return newton(F)
    
lindsay = entr_func(lindsay, lindsay_der, lindsay_hess, lindsay_conj, lindsay_conj_der, lindsay_prox, lambda a: 1/(1 - a))


def perimeter(x, a):
    if a == 1:
        return jensen_shannon(x, a)
    elif a == 0:
        return 1/2*tv(x, a)
    else:
        return np.choose(x >= 0, [np.inf, np.sign(a) / (1 - a) * ( (x**(1/a) + 1)**a - 2**(a - 1) * (x + 1) )])
        
def perimeter_der(x, a):
    if a == 1:
        return jensen_shannon_der(x, a)
    elif a == 0:
        return 1/2*tv_der(x, a)
    else:
        return np.choose(x > 0, [np.inf, np.sign(a) / (1 - a) * ( (x**(-1/a) + 1)**(a - 1) - 2**(a - 1) )])
        
        
def perimeter_conj(x, a):
    h_a = (1 - a) / np.sign(a) * x + 2**(a - 1)
    return np.choose(x < rec_const(perimeter, a), [np.inf,
    x - np.sign(a) / (1 - a) * (h_a**(a/(a-1)) - 2**(a - 1)) * (h_a**(1/(a - 1)) - 1)**(-a) + np.sign(a)/(1 - a) * 2**(a - 1)])
# TODO: implement perimeter_conj_der

def reverse_kl(x, a):
    tol = 1e-30
    return np.choose(x > 0, [np.inf, x - 1 - np.log(x + tol)])

def reverse_kl_der(x, a):
    return np.choose(x > 0, [np.inf, (x-1)/x])
    
def reverse_kl_hess(x, a):
    return 1 / x**2
    
def reverse_kl_conj(x, a):
    return np.choose(x < 1, [np.inf, - np.log(1 - x) ])
    
def reverse_kl_conj_der(x, a):
    return np.choose(x < 1, [np.inf, 1/(1 - x) ])  
    
def reverse_kl_prox(x, a, eta):
    return 1/2 * (x - eta + torch.sqrt( (x - eta)**2 + 4 * eta))   
    
reverse_kl = entr_func(reverse_kl, reverse_kl_der, reverse_kl_hess, reverse_kl_conj, reverse_kl_conj_der, reverse_kl_prox, 1)

def jensen_shannon(x, a):
    tol = 1e-30
    return np.choose(x > 0, [np.inf, np.log(tol + x) - (x + 1) * np.log((x+1)/2) ])

def jensen_shannon_der(x, a):
    return np.choose(x > 0, [np.inf, 1/x - 1 - np.log((x+1)/2)])
    
def jensen_shannon_hess(x, a):
    return -1 / x**2 - 1/(1 + x)
    
def jensen_shannon_conj(x, a):
    return np.choose(x < np.log(2), [np.inf, - np.log(2 - np.exp(x))])
    
def jensen_shannon_conj_der(x, a):
    return np.choose(x < np.log(2), [np.inf, np.exp(x) / (2 - np.exp(x) ) ]) 
    
def jensen_shannon_prox(x, a, eta):
     F = lambda y: torch.log(2*y) - torch.log(1 + y) + (y - x) / eta 
     return newton(F)
     
jensen_shannon = entr_func(jensen_shannon, jensen_shannon_der, jensen_shannon_hess, jensen_shannon_conj, jensen_shannon_conj_der, jensen_shannon_prox, np.log(2))

def power(x, a):
    if a != 0:
        return 1/a * tsallis(x)
    else:
        return reverse_kl(x)
        
def power_der(x, a):
    if a != 0:
        return 1/a * tsallis_der(x)
    else:
        return reverse_kl_der(x)
 
def power_conj(x, a):
    if a != 0:
        return 1/a * tsallis_conj(x / a)
    else:
        return reverse_kl_der(x)

def power_conj_der(x, a):
    if a != 0:
        return 1/a**2 * tsallis_conj_der(x / a)
    else:
        return reverse_kl_conj_der(x)
        
    
def tv(x, a):
    return np.choose(x >= 0, [np.inf, np.abs(x - 1)])

def tv_der(x, a):
    return np.choose(x >= 0, [np.inf, np.sign(x - 1)])
    
def tv_hess(x, a):
    return np.choose(x >= 0, [np.inf, 0])
    
def tv_conj(y, a):
    return np.where(y <= 1, np.maximum(y, -1), np.inf)

def tv_conj_der(x, a):
    return np.select([np.abs(x) <= 1], [1], default=0)
  
def tv_prox(x, a, eta):
    return torch.where(x + eta >= 0, torch.nn.Softshrink(lambd=eta)(x-1), 0)
    
tv = entr_func(tv, tv_der, tv_hess, tv_conj, tv_conj_der, tv_prox, 1)
    
def matusita(x, a):
    return np.abs(1 - x**(a))**(1/a)
    
def matusita_der(x, a):
    return x**(a-1) * (x**a - 1) * np.abs(1 - x**a)**(1/a - 2)
    
def matusita_conj(x, a):
    return np.choose(x < 1, [np.inf, 
    (x - np.abs(x)**(1/(1 - a))) * (1 - np.sgn(x) * np.abs(x)**(a/(1 - a)))**(-1/a)])
# TODO: implement matusita_conj_der

def kafka(x, a):
    return np.abs(1 - x)**(1 / a) * (1 + x)**(1 - 1 / a)
    
def kafka_der(x, a):
    return 1 / a * ((x - 1) * (a * (x - 1) + 2) * (x + 1)**(-1/a) * np.abs(x - 1)**(1/a - 2))

# see: math.stackexchange.com/a/4833038/545914
def kafka_conj(x, a):
    assert 0 <= x <= 1
    return 2 * (x - sp.special.betaincinv(1/a - 1, 2, x)**(1/a))/(sp.special.betaincinv(2, 1/a - 1, 1 - x)) - x  # this is only valid for 0 <= x <= 1

# TODO: implement kafka_conj, kafka_conj_der
    
def marton(x, a):
    return reLU(1 - x)**2
    
def marton_der(x, a):
    return 2*reLU(1 - x)
    
def marton_hess(x, a):
    return 0
    
def marton_conj(x, a):
    return np.choose(x <= 0, [np.inf, np.choose(x <= -2, [-1, 1/4*x**2 + x])])
    
def marton_conj_der(x, a):
    return np.choose(x <= 0, [np.inf, np.choose(x <= -2, [0, 1/2*x + 1])])
    
def marton_prox(x, a, eta):
    return torch.where(x >= - 2 * eta, torch.where(x <= 1, (x + 2 * eta)/(1 + 2*eta), x), 0)
        
marton = entr_func(marton, marton_der, marton_hess, marton_conj, marton_conj_der, marton_prox, 0)

def eq(x, a):
    return torch.where(x == 1, 0, float('inf'))
    
def eq_der(x, a):
    return eq(x, a)

def eq_hess(x, a):
    return eq(x, a)
    
def eq_conj(x, a):
    return x
    
def eq_conj_der(x, a):
    return 1
    
def eq_prox(x, a)
    return 1/2 * (1 - x)**2
    
equality_indicator = entr_func(eq, eq_der, eq_hess, eq_conj, eq_conj_der, eq_prox, float('inf'))

# define recession constants
def rec_const(div, a = None):
    if div == 'power':
        if a >= 1:
            return float('inf')
        elif a < 1 and a != 0:
            return 1 / (1 - a)
        elif a == 0:
            return 1

    if div in ['jeffreys', 'chi']:
        return float('inf')
     
    if div == 'lindsay':
        return 1/(1 - a)
        
            
    if div in ['tv', 'reverse_kl', 'matusita', 'kafka']:
        return 1
    
        
    if div == 'perimeter':
        if a > 0:
            return 1/(1 - a) * (1 - 2**(a - 1))
        elif a < 0:
            return 1/(1 - a) * 2**(a - 1)
        elif a == 0:
            return 1/2
        elif a == 1:
            return np.log(2)        