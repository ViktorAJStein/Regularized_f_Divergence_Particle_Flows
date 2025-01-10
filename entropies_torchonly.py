import torch
import numpy as np
import scipy as sp
# from torchlambertw.special import lambertw as lw

torch.set_default_dtype(torch.float64)  # set higher precision
my_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def prox_finder(entr, a, x, eta):
    pertub = lambda y: entr(y, a) + 1/(2*eta) * (y - x)**2
    return minimize_strictly_convex_adaptive_vectorized(pertub, x)
    
def minimize_strictly_convex_adaptive_vectorized(f, x0, alpha=1.0, tol=1e-4, max_iter=25, increase_factor=1.5, decrease_factor=0.5):
    """
    Vectorized version of the derivative-free optimization algorithm for strictly convex functions.

    Args:
        f (callable): The objective function to minimize, applied element-wise to tensors.
        x0 (torch.Tensor): Initial guesses (1D tensor).
        alpha (float): Initial step size.
        tol (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.
        increase_factor (float): Factor to increase step size (default 1.5).
        decrease_factor (float): Factor to decrease step size (default 0.5).

    Returns:
        torch.Tensor: Approximate minimizers.
        torch.Tensor: Function values at the minimizers.
    """
    x = x0.clone()  # Copy initial guesses
    step_size = torch.full_like(x, x.new_tensor(alpha))  # Initialize step sizes
    converged = torch.zeros_like(x, dtype=torch.bool, device=x.device)  # Convergence flags
    for k in range(max_iter):
        f_current = f(x)

        # Test both directions
        x_minus = torch.clamp(x - step_size, min=0)  # Ensure non-negativity
        x_plus = x + step_size

        f_minus = f(x_minus)
        f_plus = f(x_plus)

        # Determine the best move for each element
        move_minus = (f_minus < f_current) & (f_minus <= f_plus)
        move_plus = (f_plus < f_current) & (f_plus < f_minus)

        # Update positions based on the best move
        x_new = torch.where(move_minus, x_minus, torch.where(move_plus, x_plus, x))

        # Adjust step size
        improvement = move_minus | move_plus
        step_size = torch.where(improvement, step_size * increase_factor, step_size * decrease_factor)

        # Update convergence flags
        converged = converged | ((~improvement) & (step_size < tol))

        # Stop early if all have converged
        if converged.all():
            break

        # Update x
        x = x_new
    return x


def reLU(x):
    return 1/2*(x + torch.abs(x))

    
    
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
 
       
# div.prox takes inputs of the type (torch.Tensor, torch.double, torch.double)
# todo: state this for the other functions as well

'''
def prox_finder(entr, a, eta, x):
    def obj(y, eta):
        return entr.fnc(y)  + 1/(2*eta) * torch.norm(x - y)**2
    return my_L_BFGS_B(x, obj, low=torch.tensor([0], device=my_device), high = torch.tensor([1e10], device=my_device))
    
def tight_dual(entr, a,  eta, h):
    def obj(lambd):
        return 1 / h.shape[0] * torch.sum(entr.conj(h + lambd)) - lambd
    return my_L_BFGS_B(x, obj, low = torch.tensor([-float('inf')], device=my_device), high = (torch.max(h) + entr.reces) * torch.ones(h.shape[0], device=my_device))
'''   

# the conjugate f_a* of the entropy function f_a for a > 1
def tsallis_conj(x, a):
    if a != 1:
        return reLU((a - 1)/a * x + 1)**(a / (a - 1)) - 1
    else:
        return torch.exp(x) - 1
   
def tsallis_conj_der(x, a):
    if a != 1:
        return reLU( (a - 1)/a * x + 1)**(1/(a - 1))
    else:
        return torch.exp(x)

def tsallis(x, a):
    if a != 1:
        return torch.where(x >= 0, (x**a - a*x + a - 1)/(a - 1), torch.tensor(float("Inf"), device=x.device))
    else:
       return torch.xlogy(x, x) - x + 1
   
def tsallis_der(x, a):
    if a != 1:
        return torch.where(x >= 0, a / (a - 1) * ( (x+1e-30)**(a - 1) - 1), float("Inf"))
    else:
        return torch.log(x+1e-30)
        
def tsallis_hess(x, a):
    if a != 1:
        return 1.0/x
    else:
        return a * x**(a - 2)

        
def tsallis_prox(x, a, eta):
    '''
    @inputs: a tensor and two floats, (torch.Tensor, torch.double, torch.double)
    '''
    if a == 2:
        return reLU(x + 2 * eta) / (2 * eta + 1)
    elif a == 1:
        return eta * lw(1/eta * torch.exp(x/eta))
    else:
        return prox_finder(tsallis, a, x, eta)

def tsallis_rec(a):
    if a >= 1:
        return float('inf')
    elif 0 < a < 1:
        return a / (1 - a)
        
tsa = entr_func(tsallis, tsallis_der, tsallis_hess, tsallis_conj, tsallis_conj_der, tsallis_prox, tsallis_rec)
                
def jeffreys(x, a):
    return torch.xlogy(x - 1, x)

def jeffreys_der(x, a):
    return (x - 1)/x + torch.log(x)
    
def jeffreys_hess(x, a):
    return (x - 2) / (x - 1)**2
    
def jeffreys_conj(x, a):
    lambert = lw(torch.exp(1 - x))
    return x - 2 + lambert + 1/lambert

def jeffreys_conj_der(x, a):
    return 1 / lw(torch.exp(1 - x))

def jeffreys_prox(x, a, eta):
    return prox_finder(jeffreys, a, eta, x)

    
jeffreys = entr_func(jeffreys, jeffreys_der, jeffreys_hess, jeffreys_conj, jeffreys_conj_der, jeffreys_prox, lambda a: float('inf'))

  
def chi(x, a):
    assert a > 1
    return torch.where(x >= 0, torch.abs(x - 1) ** a, torch.tensor(float("Inf"), device=x.device))

def chi_der(x, a):
    assert a > 1
    return torch.where(x >= 0, a * torch.abs(x - 1)**(a - 1) * np.sign(x - 1), torch.tensor(float("Inf"), device=x.device))

def chi_conj(x, a):
    assert a > 1
    result = torch.where(x >= -a, x + (a - 1) * torch.exp(a/(a-1)*torch.log(torch.abs(x)/a+1e-10)), - torch.ones(x.shape, device=x.device))
    if torch.isinf(result).sum():
        raise Exception('chi_conj returned inf values')
    return result
    
def chi_conj_der(x, a):
    assert a > 1
    return torch.where(x >= -a, 1 + (a)**(1/(1-a)) * torch.abs(x)**(1/(a - 1)) * np.sign(x), torch.zeros(x.shape, device=x.device))

def chi_prox(x, a, eta):
    return prox_finder(chi, a, eta, x)

chi_entr = entr_func(chi, chi_der, None, chi_conj, chi_conj_der, chi_prox, lambda a: float('inf'))

  
# divergences with non-finite conjugates    
def lindsay(x, a):
    assert a >= 0 and a <= 1
    return (x - 1)**2 / (a + (1 - a)*x)
    
def lindsay_der(x, a):
    return ((1 - x) * (-a + (a - 1) * x - 1)) / ((a - 1) * x - a)**2
    
def lindsay_hess(x, a):
    return 2 / (a * (1 - x) + x)**3
    
def lindsay_conj(x, a):
    return torch.where(x <= 1/(1 - a), ( a*(a - 1)*x - 2*torch.sqrt( (a - 1)*x+1 ) + 2 )/(a - 1)**2, torch.tensor(float("Inf")))
    
def lindsay_conj_der(x, a):
    return torch.where(x < 1/(1 - a), 1 / (a - 1) * (a - 1 / torch.sqrt( (a - 1) * x + 1 )), torch.tensor(float("Inf"), device=x.device))
    
def lindsay_prox(x, a, eta):
    return prox_finder(lindsay, a, eta, x)
    
lind = entr_func(lindsay, lindsay_der, lindsay_hess, lindsay_conj, lindsay_conj_der, lindsay_prox, lambda a: 1/(1 - a))


def perimeter(x, a):
    if a == 1:
        return js(x, a)
    elif a == 0:
        return 1/2*tv(x, a)
    else:
        return torch.where(x >= 0, np.sign(a) / (1 - a) * ( (x**(1/a) + 1)**a - 2**(a - 1) * (x + 1) ), torch.tensor(float("Inf"), device=x.device))
        
def perimeter_der(x, a):
    if a == 1:
        return js_der(x, a)
    elif a == 0:
        return None
    else:
        return torch.where(x > 0, np.sign(a) / (1 - a) * ( (x**(-1/a) + 1)**(a - 1) - 2**(a - 1) ), torch.tensor(float("Inf"), device=x.device))
        
def perimeter_conj(x, a):
    h_a = (1 - a) / np.sign(a) * x + 2**(a - 1)
    return torch.where(x < rec_const('perimeter', a), x - np.sign(a) / (1 - a) * (h_a**(a/(a-1)) - 2**(a - 1)) * (h_a**(1/(a - 1)) - 1)**(-a) + np.sign(a)/(1 - a) * 2**(a - 1), torch.tensor(float("Inf"), device=x.device))

def perimeter_prox(x, a, eta):
    return prox_finder(perimeter, a, eta, x)
    
def perimeter_rec_const(a):
    if a > 0.0:
        return 1.0/(1.0 - a) * (1.0 - 2**(a - 1.0))
    elif a < 0.0:
        return 1.0/(1.0 - a) * 2**(a - 1.0)
    elif a == 0.0:
        return 0.5
    elif a == 1.0:
        return torch.log(2) 
        
per = entr_func(perimeter, perimeter_der, None, perimeter_conj, None, perimeter_prox, perimeter_rec_const)


def burg(x, a):
    tol = 1e-30
    return torch.where(x > 0, x - 1 - np.log(x + tol), torch.tensor(float("Inf"), device=x.device))

def burg_der(x, a):
    return torch.where(x > 0, (x-1)/x, torch.tensor(float("Inf"), device=x.device))
    
def burg_hess(x, a):
    return 1 / x**2
    
def burg_conj(x, a):
    return torch.where(x < 1, - torch.log(1 - x), torch.tensor(float("Inf"), device=x.device))
    
def burg_conj_der(x, a):
    return torch.where(x < 1, 1/(1 - x), torch.tensor(float("Inf"), device=x.device))  
    
def burg_prox(x, a, eta):
    return 1/2 * (x - eta + torch.sqrt( (x - eta)**2 + 4 * eta))   
    
reverse_kl = entr_func(burg, burg_der, burg_hess, burg_conj, burg_conj_der, burg_prox, lambda a: 1)


def js(x, a):
    tol = 1e-30
    return torch.where(x > 0, torch.log(tol + x) - (x + 1) * torch.log((x+1)/2), torch.tensor(float("Inf"), device=x.device))

def js_der(x, a):
    return torch.where(x > 0, 1/x - 1 - torch.log((x+1)/2), torch.tensor(float("Inf"), device=x.device))
    
def js_hess(x, a):
    return - 1 / x**2 - 1/(1 + x)
    
def js_conj(x, a):
    return torch.where(x < torch.tensor(np.log(2)), - torch.log(torch.tensor(np.log(2)) - torch.exp(x)), torch.tensor(float("Inf"), device=x.device))
    
def js_conj_der(x, a):
    return torch.where(x < torch.tensor(np.log(2)), torch.exp(x) / (torch.tensor(np.log(2)) - torch.exp(x) ), torch.tensor(float("Inf"), device=x.device)) 
    
def js_prox(x, a, eta):
    return prox_finder(js, a, eta, x)
     
jensen_shannon = entr_func(js, js_der, js_hess, js_conj, js_conj_der, js_prox, lambda a: torch.tensor(np.log(2)))

# TODO: convert them to torch!
'''
def power(x, a):
    if a != 0:
        return 1/a * tsallis(x)
    else:
        return burg(x)
        
def power_der(x, a):
    if a != 0:
        return 1/a * tsallis_der(x)
    else:
        return burg_der(x)
 
def power_conj(x, a):
    if a != 0:
        return 1/a * tsallis_conj(x / a)
    else:
        return burg_der(x)

def power_conj_der(x, a):
    if a != 0:
        return 1/a**2 * tsallis_conj_der(x / a)
    else:
        return burg_conj_der(x)
        
    
def matusita(x, a):
    return np.abs(1 - x**(a))**(1/a)
    
def matusita_der(x, a):
    return x**(a-1) * (x**a - 1) * np.abs(1 - x**a)**(1/a - 2)
    
def matusita_conj(x, a):
    return torch.where(x < 1, [np.inf, 
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
''' 

    
def tv(x, a):
    return torch.where(x >= 0, torch.abs(x - 1), torch.tensor(float("Inf"), device=x.device))
    
def tv_conj(y, a):
    return torch.where(y <= 1, torch.max(y, torch.tensor(-1.0)), torch.tensor(float("Inf"), device=y.device))
  
def tv_prox(x, a, eta):
    return torch.where(x + eta >= 0, torch.nn.Softshrink(lambd=eta)(x-eta), 0)
    
tv = entr_func(tv, None, None, tv_conj, None, tv_prox, lambda a: 1)
  
def marton(x, a):
    return reLU(1 - x)**2
    
def marton_der(x, a):
    return 2*reLU(1 - x)
    
def marton_conj(x, a):
    return torch.where(x <= 0, torch.where(x <= -2, -1, 1/4*x**2 + x), torch.tensor(float("Inf"), device=x.device))
def marton_prox(x, a, eta):
    return torch.where(x >= - 2 * eta, torch.where(x <= 1.0, (x + 2 * eta)/(1 + 2*eta), x), 0.0)

marton = entr_func(marton, marton_der, None, marton_conj, None, marton_prox, lambda a: 0)


def eq(x, a):
    return torch.where(x == 1.0, 0.0, float('inf'))
    
def eq_conj(x, a):
    return x   
    
def eq_prox(x, a, eta):
    return torch.ones(x.shape, device=x.device)
    
equality_indicator = entr_func(eq, None, None, eq_conj, None, eq_prox, lambda a: float('inf'))

def z(x, a):
    return torch.where(x >= 0.0, 0.0, float('inf'))
    
def z_conj(x, a):
    return torch.where(x < 0.0, 0.0, float('inf'))
    
def z_prox(x, a, eta):
    return reLU(x)
    
zero = entr_func(z, None, None, z_conj, None, z_prox, lambda a: float('inf'))

# define recession constants
def rec_const(div, a = None):
    if div == 'power':
        if a >= 1.0:
            return float('inf')
        elif a < 1.0 and a != 0:
            return 1.0 / (1.0 - a)
        elif a == 0.0:
            return 1.0

    if div in ['jeffreys', 'chi']:
        return float('inf')
     
    if div == 'lindsay':
        return 1.0/(1.0 - a)
        
            
    if div in ['tv', 'burg', 'matusita', 'kafka']:
        return 1.0
    
        
    if div == 'perimeter':
        if a > 0.0:
            return 1.0/(1.0 - a) * (1.0 - 2**(a - 1.0))
        elif a < 0.0:
            return 1.0/(1.0 - a) * 2**(a - 1.0)
        elif a == 0.0:
            return 0.5
        elif a == 1.0:
            return np.log(2)  
                