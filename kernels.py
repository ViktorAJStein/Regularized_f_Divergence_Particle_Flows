import torch
import numpy as np

'''
This auxilliary file provides functions for many kernels, including Gaussian, (inverse) multiquadric,
Matern and many more, as well as their derivatives with respect to the first argument, x.
'''

# Gaussian kernel with width s
def gauss(x, y, s):
    return (-1 / (2*s) * (x - y) ** 2).sum(axis=-1).exp()

# derivative of Gaussian kernel    
def gauss_der(x, y, s):
    diff = y[:,None, :] - x[None,:, :]
    return 1 / s * (-1 / (2*s) * torch.linalg.vector_norm(diff, dim=2, keepdim=True)**2).exp() * diff
    

def imq(x, y, s):
    return (s + (((x - y) ** 2)).sum(axis=-1)) ** -(1/2)
    
def imq_der(x, y, s):
    diff = y[:,None, :] - x[None,:, :]
    pref = (torch.linalg.vector_norm(diff, dim=2, keepdim=True)**2 + s) ** -(3/2)
    return pref * diff
 
    
def matern(x, y, s): # nu = 3/2
    r = ((x - y) ** 2).sum(axis=-1)
    return (1 + torch.sqrt(3*r) / s) * (- torch.sqrt(3*r) / s).exp()
    #2**(1 - nu) / sp.special.gamma(nu) * (np.sqrt(2 * nu * r) / s)**nu * sp.special.kv(nu, np.sqrt(2 * nu * r)/ s)
    
def matern_der(x, y, s): # nu = 3/2
    diff = y[:,None, :] - x[None,:, :]
    r = torch.linalg.vector_norm(diff, dim=2, keepdim=True)
    return 3/s**2 * (- 3 / s * r).exp() * diff
    
    
def matern2(x, y, s): # nu = 5/2
    r = ((x - y) ** 2).sum(axis=-1)
    return (1 + torch.sqrt(5*r) / s + 5*r/(3*s**2) ) * (- torch.sqrt(5*r) / s).exp()

def matern2_der(x, y, s): # nu = 5/2
    diff = y[:,None, :] - x[None,:, :]
    r = torch.linalg.vector_norm(diff, dim=2, keepdim=True)**2
    return 5/6 * 1/s**3 * (torch.sqrt(5*r) + s) * (- torch.sqrt(5*r) / s).exp() * diff
    
    
def compact(x, y, q): # this expression depends on the dimension of the data points being d = 2 (or more generally, that floor(d/2) = 1)
    r = torch.sqrt(((x - y) ** 2).sum(axis=-1))
    return torch.nn.functional.relu(1 - r)**(q + 2)
    
def compact_der(x, y, s):
    diff = y[:,None, :] - x[None,:, :]
    r = torch.linalg.vector_norm(diff, dim=2, keepdim=True)
    return diff/r* (s + 2) *torch.nn.functional.relu(1 - r)**(s + 1)
    

def compact2(x, y, s):  # this expression depends on the dimension of the data points being d = 2
    r = torch.sqrt(((x - y) ** 2).sum(axis=-1))
    return torch.nn.functional.relu(1 - r)**(s + 3) * ( (s + 3)*r + 1 ) 
    
def compact2_der(x, y, s):
    diff = y[:,None, :] - x[None,:, :]
    r = torch.linalg.vector_norm(diff, dim=2, keepdim=True)
    return 1/2 * diff* (s + 3) * (s + 4) * torch.nn.functional.relu(1 - r)**(s + 2)

    
def inv_quad(x, y, s):
    r2 = ((x - y) ** 2).sum(axis=-1)
    return 1/(1 + s*r2)
    
def inv_quad_der(x, y, s):
    diff = y[:,None, :] - x[None,:, :]
    r = torch.linalg.vector_norm(diff, dim=2, keepdim=True)
    return 2*s/(1 + s*r**2)**2 * diff
    

def inv_log(x, y, s, beta = -1/2):
    return (s + torch.log( 1 + ((x - y)**2).sum(axis=-1) ) )**(beta)
 
def inv_log_der(x, y, s, beta=-1/2):
    diff = y[:,None, :] - x[None,:, :]
    r = torch.linalg.vector_norm(diff, dim=2, keepdim=True)
    prefactor = - 2*beta/ (1 + r) * (s + torch.log(1 + r))**(beta - 1)
    return prefactor * diff 
        

def student(x, y, s):
    s = torch.tensor([s], device='cuda')
    prefactor = torch.special.gammaln( (s + 1)/2 ).exp() / torch.sqrt(torch.pi * s) * 1/torch.special.gammaln(s/2).exp()
    return prefactor * (1 + (((x - y) ** 2)).sum(axis=-1) / s) ** (-1/2*(s + 1))
    
def student_der(x, y, s):
    s = torch.tensor([s], device='cuda')
    prefactor = torch.special.gammaln((s + 1)/2).exp() / torch.sqrt(torch.pi * s) * 1/torch.special.gammaln(s/2).exp()
    diff = y[:,None, :] - x[None,:, :]
    pref = prefactor * (1 + 1/s) * (1 + torch.linalg.vector_norm(diff, dim=2, keepdim=True)**2 / s) ** (-1/2*(s + 3))
    return pref * diff

   
def emb_const(kern, s):
    # returns embedding constant of H_K \hookrightarrow C_0
    if kern in [gauss, matern, matern2, compact, compact2, laplace, multiquad, sinc, inv_quad]:
        return 1
    elif kern in [imq, inv_log]:
        return np.sqrt(s**(1/4)) 
    elif kern == logistic:
        return np.sqrt(1/s)
    elif kern == student:
        s = torch.tensor([s])
        prefactor = torch.special.gammaln( (s + 1)/2 ).exp() / torch.sqrt(torch.pi * s) * 1/torch.special.gammaln(s/2).exp()
        return prefactor.item()

# see Ex. 4 in Modeste, Dombry: https://hal.science/hal-03855093
# These kernels metrizes the W2-metric,
# but are not differentiable, not translation-invariant and not bounded   
def W2_1(x, y, s):
    return gauss(x, y, s) + (torch.abs(x) * torch.abs(y)).sum(axis=-1)

def W2_1_der(x, y, s):
    return gauss_der(x, y, s) + torch.sign(x[None, :, :]) * torch.abs(y[:, None, :])

def W2_2(x, y, s):
    return gauss(x, y, s) + (x**2 * y**2).sum(axis=-1)

def W2_2_der(x, y, s):
    return gauss_der(x, y, s) + 2*x[None, :, :]*y[:, None, :]*y[:, None, :]
    
# these kernels below do not yield sensible results
# there are multiple reasons, i.e the thin plate spline
# is not positive definite, sinc is not universal

# inspired by KALE code from https://github.com/pierreglaser/kale-flow/tree/master    
def energy(x, y, s):
    eps = 1e-8
    xx0 = ( (x**2).sum(axis=-1) + eps) ** (s / 2)
    yx0 = ( (y**2).sum(axis=-1) + eps) ** (s / 2)
    xy = ( ((x-y)**2).sum(axis=-1) + eps) ** (s / 2)
    # pretending eps = 0, this is 1/2 * (|| x ||^s + || y ||^s - || x - y ||^s)
    return 0.5 * (xx0 + yx0 - xy)
    
def energy_der(x, y, s):
    eps = 1e-8
    new_y = y[:,None, :]
    diffyx = new_y - x[None,:, :]
    diffx = torch.zeros_like(new_y) - x[None,:, :]
    ryx2 = torch.linalg.vector_norm(diffyx, dim=2, keepdim=True)**2
    rx2 = torch.linalg.vector_norm(diffx, dim=2, keepdim=True)**2
    px0 = (rx2 + eps) ** (s / 2 - 1)
    pyx = (ryx2 + eps) ** (s / 2 - 1)
 
    return s/2 * ( px0*x + pyx*diffyx)
    

def thin_plate_spline(x, y, s):
    tol=1e-16
    r = ((x - y) ** 2).sum(axis=-1)**(1/2)
    return r * torch.log(r**r + tol)

def thin_plate_spline_der(x, y, s):
    tol=1e-16
    diff = y[:,None, :] - x[None,:, :]
    r = torch.linalg.vector_norm(diff, dim=2, keepdim=True)
    return - 1/2 * diff * (torch.log(r**2 + tol) + 1)

# not radial    
def squared_dot(x, y, s):
    return 1/2*torch.dot(x,y)**2

def squared_dot(x, y, s):
    return y

#not universal    
def sinc(x, y, s):
    r = ((x - y) ** 2).sum(axis=-1)**(1/2)
    return torch.sin(s*r)/r
    
def sinc_der(x, y, s):
    diff = y[:,None, :] - x[None,:, :]
    r = torch.linalg.vector_norm(diff, dim=2, keepdim=True)
    return ( s * torch.cos(s * r) / r**2 - torch.sin(s*r) / r**(3/2) ) * diff

# not positive definite
def multiquad(x, y, s):
    r2 = ((x - y) ** 2).sum(axis=-1)
    return torch.sqrt(1 + s*r2)
    
def multiquad_der(x, y, s):
    diff = y[:,None, :] - x[None,:, :]
    r = torch.linalg.vector_norm(diff, dim=2, keepdim=True)
    return - s/torch.sqrt(1 + s*r**2) * diff
    

# not differentiable at x = y    
def laplace(x, y, s):
    return (-1 / s * (x - y).abs()).sum(axis=-1).exp()
    
def laplace_der(x, y, s):
    diff = y[:,None, :] - x[None,:, :]
    r = torch.linalg.vector_norm(diff, dim=2, keepdim=True)
    return 1/(r * s) * (-r/s).exp() * diff

    
# not differentiable at x = y 
def logistic(x, y, s):
    r = (x - y).abs()
    expp = (-1 / s * r).sum(axis=-1).exp()
    return  expp / (s * (1 + expp)**2)
    
def logistic_der(x, y, s):
    diff = y[:,None, :] - x[None,:, :]
    r = torch.linalg.vector_norm(diff, dim=2, keepdim=True)
    expp = (1 / s * r).sum(axis=-1).exp()
    return expp * (expp - 1) / (s**2 * (expp + 1)**3) * diff / r
