import numpy as np
import torch


def generate_prior_target(N, st, target):
    if target == 'circles':
        return generate_circles(N, st=st)
    elif target == 'cross':
        return generate_cross(N, st=st)
    elif target == 'bananas':
        return generate_bananas(N, st=st)


def neals_funnel(n_samples, st=314):
    # Generate samples from Neal's funnel
    rs = np.random.RandomState(st)
    y = rs.normal(0, 2, size=n_samples)
    x = rs.normal(0, np.exp(y/3), size=n_samples)
    y = (y+7.5)
    return np.column_stack((x, y))


def generate_circles(N, st=42, r=.3, delta=.5):
    '''
    1. Generate three rings target, each ring has N points sampled uniformly.
    The rings have a radius of r and a separation of _delta.

    2. Generate prior, which is a Gaussian with very small variance centered at
    leftmost point of the rightmost ring.
    Returns
    -------
    X : np.array, shape = (N, 2)
        target.
    Y : np.array, shape = (N, 2)
        prior.

    '''
    n = int(N/3)
    # TODO: convert to pytorch code
    X = np.c_[r * np.cos(np.linspace(0, 2 * np.pi, n + 1)), r * np.sin(np.linspace(0, 2 * np.pi, n + 1))][:-1]  # noqa
    for i in [1, 2]:
        X = np.r_[X, X[:n, :]-i*np.array([0, (2 + delta) * r])]
    rs = np.random.RandomState(st)
    Y = rs.standard_normal((n*(2+1), 2)) / 100 - np.array([0, r])

    Y = torch.from_numpy(Y).to(torch.float64)
    X = torch.from_numpy(X).to(torch.float64)
    return X, Y


def generate_bananas(N, st, d=2):
    u = int(N/2)

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

    # mean and variance of prior distribution
    m_p = torch.tensor([0, 4.0])
    v_p = 1 / 2000 * torch.eye(d)

    # Draw samples from the normal distribution
    multivariate_normal = torch.distributions.MultivariateNormal(m_p, v_p)
    prior = multivariate_normal.sample((N,))

    return target, prior


def generate_cross(N, st, d=2):
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

    # Draw samples from the prior normal distribution
    multivariate_normal = torch.distributions.MultivariateNormal(m_p, v_p)
    prior = multivariate_normal.sample((N,))

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
