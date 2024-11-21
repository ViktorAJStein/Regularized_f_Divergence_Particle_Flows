import torch


def armijo_search(func, q, eta, p, fx, t, tau, max_iter=20):
    for j in range(max_iter):
        if func(q + eta * p) <= fx - eta * t:
            return eta
        eta *= tau
    return eta
    

def two_way_backtracking_line_search(f, grad_f, x, direction, c1=1e-4, c2=0.9, alpha0=1.0):
    """
    Two-way backtracking line search in PyTorch.
    
    Args:
        f (callable): Objective function. Takes a tensor `x` as input and returns a scalar.
        grad_f (callable): Gradient of the objective function. Takes a tensor `x` and returns a tensor.
        x (torch.Tensor): Current point.
        direction (torch.Tensor): Descent direction.
        c1 (float): Armijo (sufficient decrease) parameter. Default is 1e-4.
        c2 (float): Wolfe (curvature) parameter. Default is 0.9.
        alpha0 (float): Initial step size. Default is 1.0.
    
    Returns:
        alpha (float): Step size satisfying the two-way backtracking conditions.
    """
    alpha = alpha0
    phi_0 = f(x)
    grad_phi_0 = grad_f(x)
    phi_prime_0 = torch.dot(grad_phi_0, direction)

    assert phi_prime_0 < 0, "Direction must be a descent direction."

    while True:
        # Evaluate the candidate step
        x_next = x + alpha * direction
        phi_next = f(x_next)
        
        # Check Armijo condition
        if phi_next > phi_0 + c1 * alpha * phi_prime_0:
            alpha /= 2  # Reduce step size
        else:
            grad_phi_next = grad_f(x_next)
            phi_prime_next = torch.dot(grad_phi_next, direction)
            
            # Check Wolfe condition
            if phi_prime_next < c2 * phi_prime_0:
                alpha *= 2  # Increase step size
            else:
                break  # Both conditions satisfied
    
    return alpha