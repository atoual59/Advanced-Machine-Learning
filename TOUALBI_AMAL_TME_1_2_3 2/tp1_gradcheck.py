import torch
from tp1 import mse, linear


# Test du gradient de MSE

yhat = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10,5, requires_grad=True, dtype=torch.float64)

print(torch.autograd.gradcheck(mse, (yhat, y)))
print(torch.autograd.gradcheck(linear, (yhat, y)))

#  TODO:  Test du gradient de Linear (sur le même modèle que MSE)

