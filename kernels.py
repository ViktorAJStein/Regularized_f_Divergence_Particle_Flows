import torch
import numpy as np
# Gaussian kernel with width s
def gauss(x, y, s):
    # if isinstance(x, torch.Tensor):
    return (-1 / (2*s) * (x - y) ** 2).sum(axis=-1).exp()
    # else:
    #     return np.exp((-1 / (2*s) * (x - y) ** 2).sum(axis=-1))

# derivative of Gaussian kernel
def gauss_der(x, y, s):
    diff = y[:, None, :] - x[None, :, :]
    return 1 / s * (-1 / (2*s) * torch.linalg.vector_norm(diff, dim=2, keepdim=True)**2).exp() * diff


def IMQ(x, y, s):
    return (s + (((x - y) ** 2)).sum(axis=-1)) ** -(1/2)
    
def IMQ_der(x, y, s):
    diff = y[:, None, :] - x[None, :, :]
    pref = (torch.linalg.vector_norm(diff, dim=2, keepdim=True)**2 + s) ** -(3/2)
    return pref * diff
    
def Matern(x, y, sigma): # nu = 3/2
    r = ((x - y) ** 2).sum(axis=-1)
    return (1 + torch.sqrt(3*r) / sigma) * (- torch.sqrt(3*r) / sigma).exp()
    #2**(1 - nu) / sp.special.gamma(nu) * (np.sqrt(2 * nu * r) / sigma)**nu * sp.special.kv(nu, np.sqrt(2 * nu * r)/ sigma)
    
def Matern_der(x, y, sigma): # nu = 3/2
    diff = y[:, None, :] - x[None, :, :]
    r = torch.linalg.vector_norm(diff, dim=2, keepdim=True)
    return 3/sigma**2 * (- 3 / sigma * r).exp() * diff
    
def Matern2(x, y, sigma): # nu = 5/2
    r = ((x - y) ** 2).sum(axis=-1)
    return (1 + torch.sqrt(5*r) / sigma + 5*r/(3*sigma**2) ) * (- torch.sqrt(5*r) / sigma).exp()

def Matern2_der(x, y, sigma): # nu = 5/2
    diff = y[:, None, :] - x[None, :, :]
    r = torch.linalg.vector_norm(diff, dim=2, keepdim=True)**2
    return 5/6 * 1/sigma**3 * (torch.sqrt(5*r) + sigma) * (- torch.sqrt(5*r) / sigma).exp() * diff
    
def compact(x, y, q): # this expression depends on the dimension of the data points being d = 2 (or more generally, that floor(d/2) = 1)
    r = torch.sqrt(((x - y) ** 2).sum(axis=-1))
    return torch.nn.functional.relu(1 - r)**(q + 2)
    
def compact_der(x, y, q):
    diff = y[:, None, :] - x[None, :, :]
    r = torch.linalg.vector_norm(diff, dim=2, keepdim=True)
    return diff/r*(q+2)/2*torch.nn.functional.relu(1 - r)**(q + 1)

def compact2(x, y, q):  # this expression depends on the dimension of the data points being d = 2
    r = torch.sqrt(((x - y) ** 2).sum(axis=-1))
    return torch.nn.functional.relu(1 - r)**(q + 3) * ( (q + 3)*r + 1 ) 
    
def compact2_der(x, y, q):
    diff = y[:, None, :] - x[None, :, :]
    r = torch.linalg.vector_norm(diff, dim=2, keepdim=True)
    return 1/2*diff*(q+3)*(q+4)*torch.nn.functional.relu(1 - r)**(q + 2)

    
def inv_quad(x, y, sigma):
    r2 = ((x - y) ** 2).sum(axis=-1)
    return 1/(1 + sigma*r2)
    
def inv_quad_der(x, y, sigma):
    diff = y[:, None, :] - x[None, :, :]
    r = torch.linalg.vector_norm(diff, dim=2, keepdim=True)
    return 2*sigma/(1 + sigma*r**2)**2 * diff

def inv_log(x, y, sigma, beta = -1/2):
    return (sigma + torch.log(1 + ((x - y)**2).sum(axis=-1)))**(beta)
 
def inv_log_der(x, y, sigma, beta=-1/2):
    diff = x[:, None, :] - y[None, :, :]
    r = torch.linalg.vector_norm(diff, dim=2, keepdim=True)
    return 2*beta/(1 + r) * (sigma + torch.log(1 + r))**(beta - 1) * diff

# copied from KALE code    
def energy_kernel(x, y, sigma):
    dim = x.shape[-1]
    x0 = torch.zeros(*([1] * (len(x.shape) - 1)), dim)

    def norm_torch_sq(z):
        ret = (z ** 2).sum(axis=-1)
        return ret

    def norm_numpy_sq(z):
        return (z ** 2).sum(axis=-1)

    eps = 1e-8

    if isinstance(x, torch.Tensor):
        pxx0 = (norm_torch_sq(x - x0) + eps) ** (sigma / 2)
        pyx0 = (norm_torch_sq(y - x0) + eps) ** (sigma / 2)
        pxy = (norm_torch_sq(x - y) + eps) ** (sigma / 2)
    elif isinstance(x, np.ndarray):
        x0 = x0.detach().numpy()
        pxx0 = (norm_numpy_sq(x - x0) + eps) ** (sigma / 2)
        pyx0 = (norm_numpy_sq(y - x0) + eps) ** (sigma / 2)
        pxy = (norm_numpy_sq(x - y) + eps) ** (sigma / 2)
    else:
        raise ValueError(f"type of x ({type(x)}) not understood")

    ret = 0.5 * (pxx0 + pyx0 - pxy)
    # pretending eps = 0, this is 1/2 * (|| x - 0 ||^sigma + || y - 0 ||^sigma - || x - y ||^sigma)
    return ret
    
def energy_kernel_der(x, y, sigma):
    dim = x.shape[-1]
    x0 = torch.zeros(*([1] * (len(x.shape) - 1)), dim)

    def norm_torch_sq(z):
        ret = (z ** 2).sum(axis=-1)
        return ret

    def norm_numpy_sq(z):
        return (z ** 2).sum(axis=-1)
    
    eps = 1e-8

    if isinstance(x, torch.Tensor):
        pxx0 = (norm_torch_sq(x - x0) + eps) ** (sigma / 2 - 1)
        pxy = (norm_torch_sq(x - y) + eps) ** (sigma / 2 - 1)
    elif isinstance(x, np.ndarray):
        x0 = x0.detach().numpy()
        pxx0 = (norm_numpy_sq(x - x0) + eps) ** (sigma / 2 - 1)
        pxy = (norm_numpy_sq(x - y) + eps) ** (sigma / 2 - 1)
    else:
        raise ValueError(f"type of x ({type(x)}) not understood")

    ret = sigma/2 * ( pxx0*x - pxy*(x- y) )


### these below do not yield sensible results, TPS is not PD

def thin_plate_spline(x, y, sigma):
    tol = 1e-16
    r = ((x - y) ** 2).sum(axis=-1)**(1/2)
    return r * torch.log(r**r + tol)

def thin_plate_spline_der(x, y, sigma):
    tol = 1e-16
    diff = x[:, None, :] - y[None, :, :]
    r = torch.linalg.vector_norm(diff, dim=2, keepdim=True)
    return 1/2*diff*(torch.log(r**2 + tol) + 1)

# not positive definite?
def multiquad(x, y, sigma):
    r2 = ((x - y) ** 2).sum(axis=-1)
    return torch.sqrt(1 + sigma*r2)

def multiquad_der(x, y, sigma):
    diff = x[:, None, :] - y[None, :, :]
    r = torch.linalg.vector_norm(diff, dim=2, keepdim=True)
    return sigma/torch.sqrt(1 + sigma*r**2) * diff