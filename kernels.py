import torch

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
