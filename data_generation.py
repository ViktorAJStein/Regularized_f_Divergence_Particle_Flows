import numpy as np
import torch
from sklearn import datasets

def generate_prior_target(N, M, st, target):
    targets = ['circles', 'cross', 'bananas', 'GMM', 'four_wells', 'circles', 'moons', 'swiss_roll_2d', 'swiss_roll_3d', 's_curve', 'annulus', 'low_dim_gaussian']
    if target in targets:
        return globals().get('generate_' + target)(N, M, st)
    else:
        raise ValueError("Invalid target specified")


def generate_low_dim_gaussian(N, M, st, d = 2):
    m1 = 1/2*torch.ones(d)
    M1 = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    v = 1/200*torch.eye(d)
    torch.manual_seed(st)
    normal1 = torch.distributions.MultivariateNormal(m1, v)
    target1 = normal1.sample((M,))
    target = torch.cat( (target1, torch.zeros(M, 8)), dim=1)
    prior = torch.cat( (torch.zeros((N, 8)), normal1.sample((N,)) ), dim=1 )
    return target, prior


def generate_four_wells(N, M, st, d = 2):
    quarterM = int(M/4)
    m1 = 1/2*torch.ones(d)
    M1 = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    m2 = torch.matmul(M1,m1)
    v = 1/200*torch.eye(d)
    torch.manual_seed(st)
    normal1 = torch.distributions.MultivariateNormal(m1, v)
    normal2 = torch.distributions.MultivariateNormal(-m1, v)
    normal3 = torch.distributions.MultivariateNormal(m2, v)
    normal4 = torch.distributions.MultivariateNormal(-m2, v)
    target1 = normal1.sample((quarterM,))
    target2 = normal2.sample((quarterM,))
    target3 = normal3.sample((quarterM,))
    target4 = normal4.sample((quarterM,))
    target = torch.cat( (target1, target2, target3, target4) )
    
    torch.manual_seed(st)
    prior = normal1.sample((N,))
    
    return target, prior
    

def generate_GMM(N, M, st, d = 2):
    # target = sum of two Gaussians
    # target and prior have the symmetry axis x = - y
    linspace = torch.linspace(-.5, .5, N).unsqueeze(1)
    prior = torch.cat( (linspace, - linspace),  dim=1)
    
    halfM = int(M/2)
    m1 = 1/2*torch.ones(d)
    v = 1/200*torch.eye(d)
    torch.manual_seed(st)
    normal = torch.distributions.MultivariateNormal(m1, v)
    target = normal.sample((halfM,))
    target = torch.cat( (target, - target) )

    return target, prior


def neals_funnel(N, M, st=314):
    # Generate samples from Neal's funnel
    rs = np.random.RandomState(st)
    y = rs.normal(0, 2, size=M)
    x = rs.normal(0, np.exp(y/3), size=M)
    y = (y+7.5)
    return np.column_stack((x, y))


def generate_circles(N, M, st=42, r=.3, delta=.5):
    '''
    1. Generate three rings target, each ring has N points sampled uniformly.
    The rings have a radius of r and a separation of delta.

    2. Generate prior, which is a Gaussian with very small variance centered at
    leftmost point of the rightmost ring.
    Returns
    -------
    prior  : np.array, shape = (N, 2)
        prior.
    target : np.array, shape = (M, 2)

    '''
    n = int(M // 3)
    # TODO: convert to pytorch code
    X = np.c_[r * np.cos(np.linspace(0, 2 * np.pi, n + 1)), r * np.sin(np.linspace(0, 2 * np.pi, n + 1))][:-1]  # noqa
    for i in [1, 2]:
        X = np.r_[X, X[:n, :]-i*np.array([0, (2 + delta) * r])]
    target = torch.from_numpy(X).to(torch.float64)
        
    torch.manual_seed(st)
    m = torch.tensor([0.0, -r])
    v = 1e-4*torch.eye(2)
    normal = torch.distributions.MultivariateNormal(m, v)
    prior = normal.sample((N,))
    
    return target, prior


def generate_bananas(N, M, st, d=2):
    u = int(M/2)

    torch.manual_seed(st)  # fix randomness

    # layer 1
    vert1 = torch.rand(u)
    hori1 = torch.rand(u)
    xs1 = torch.linspace(-1, 1, u) + vert1
    squared1 = xs1**2 + hori1

    # layer 2
    vert2 = torch.rand(u)
    hori2 = torch.rand(u)
    xs2 = torch.linspace(-1.5, 1.5, u) + vert2
    squared2 = 1/2*(xs2 - 1)**2 + hori2 - 4

    xs = torch.cat((xs1, xs2))
    squared = torch.cat((squared1, squared2))
    target = torch.stack((xs, squared)).transpose(0, 1)

    m_p = torch.tensor([0, 4.0])
    v_p = 1 / 2000 * torch.eye(d)
    norm = torch.distributions.MultivariateNormal(m_p, v_p)
    prior = norm.sample((N,))

    return target, prior


def generate_cross(N, M, st, d=2):
    rot_num = 4 # number of rotations
    samples = neals_funnel(int(M/rot_num), st=st)
    rotations = (360/rot_num)*np.arange(rot_num)
    
    new_samples = []
    for rotation in rotations:
        rotated_samples = rotate_points(samples, rotation)
        new_samples.append(rotated_samples)
    
    target = torch.from_numpy(np.concatenate(new_samples, axis=0))
    m_p = torch.zeros(d)
    v_p = 1/2000*torch.eye(d)
    norm = torch.distributions.MultivariateNormal(m_p, v_p)
    prior = norm.sample((N,))

    return target, prior


def rotate_point(point, angle):
    # rotate the point by angle (in radians)
    x, y = point
    angle = torch.tensor(angle)
    new_x = x * torch.cos(angle) - y * torch.sin(angle)
    new_y = x * torch.sin(angle) + y * torch.cos(angle)
    return (new_x, new_y)


def rotate_points(points, angle):
    return torch.tensor([rotate_point(point, angle) for point in points])
    
def generate_swiss_roll_2d(N, M, st):
    torch.manual_seed(st)
    theta = torch.sqrt(torch.rand(M)) * 4 * torch.pi # angles
    X1 = theta.cos() * theta
    X2 = theta.sin() * theta
    target = torch.stack((X1, X2), dim=1)
    
    m = torch.zeros(2)
    v = 1/200*torch.eye(2)
    torch.manual_seed(st)
    normal = torch.distributions.MultivariateNormal(m, v)
    prior = normal.sample((N,))

    return target, prior
    
    
def generate_swiss_roll_3d(N, M, st):
    target, _ = datasets.make_swiss_roll(n_samples=M, random_state=st, noise=.5)
    target = torch.from_numpy(target)
    m = torch.tensor([0.0, 10.0, -5.0])
    v = 1/200*torch.eye(3)
    torch.manual_seed(st)
    normal = torch.distributions.MultivariateNormal(m, v)
    prior = normal.sample((N,))
    return target, prior


def generate_s_curve(N, M, st):
    target, _ = datasets.make_s_curve(n_samples=M, random_state=st, noise=.1)
    target = torch.from_numpy(target)
    m = torch.zeros(3)
    v = 1/200*torch.eye(3)
    torch.manual_seed(st)
    normal = torch.distributions.MultivariateNormal(m, v)
    prior = normal.sample((N,))
    return target, prior
    
def generate_annulus(N, M, st):
    target, _ = datasets.make_circles(n_samples=M, random_state=st)
    target = torch.from_numpy(target)
    m = torch.zeros(2)
    v = 1/200*torch.eye(2)
    torch.manual_seed(st)
    normal = torch.distributions.MultivariateNormal(m, v)
    prior = normal.sample((N,))
    return target, prior

def generate_moons(N, M, st):
    target, _ = datasets.make_moons(n_samples=M, random_state=st, noise=.1)
    target = torch.from_numpy(target)
    m = 1/2*torch.tensor([2.0, 1.0])
    v = 1/200*torch.eye(2)
    torch.manual_seed(st)
    normal = torch.distributions.MultivariateNormal(m, v)
    prior = normal.sample((N,))
    
    return target, prior