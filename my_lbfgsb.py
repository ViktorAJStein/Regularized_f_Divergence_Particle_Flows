'''
import torch
from torch.optim.lbfgsb import LBFGSB 

def my_L_BFGS_B(x_init, objective, low, high):
    # x = x_init.clone().detach().requires_grad_(True)
    optimizer = LBFGSB([x], lower_bound=low, upper_bound=high)

    for step in range(20):  
        def closure():
          if torch.is_grad_enabled():
              optimizer.zero_grad()  # Clear gradients
          loss = objective(x)   # Compute the loss
          if loss.requires_grad:
              loss.backward()       # Compute gradients
          return loss      
        optimizer.step(closure)  # Perform one optimization step
    loss = objective(x)
    return x.detach(), loss.data.item()

'''
import torch
from torch.optim.lbfgsb import LBFGSB 
      
def my_L_BFGS_B(x_init, objective, low, high):
    x = x_init.clone().detach().requires_grad_(True)

    optimizer = LBFGSB([x], lower_bound = low, upper_bound = high)
    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()  # Clear gradients
        loss = objective(x)   # Compute the loss
        if loss.requires_grad:
            loss.backward()       # Compute gradients
        return loss
    for step in range(20):
        optimizer.step(closure)
    loss = closure().detach()
    return x.detach(), loss

