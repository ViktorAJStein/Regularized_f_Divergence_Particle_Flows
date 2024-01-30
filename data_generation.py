import numpy as np
import torch

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
