# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/baskiotis/venv/amal/3.7/bin/activate

import torch
from torch.autograd import Function
from torch.autograd import gradcheck



class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """Début d'implementation de la fonction MSE"""
    @staticmethod
    def forward(ctx, yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)
        return (yhat-y)**2
        #  TODO:  Renvoyer la valeur de la fonction


    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors
        #  TODO:  Renvoyer par les deux dérivées partielles (par rapport à yhat et à y)
        return grad_output*(2*(yhat-y)), grad_output*(-2*(yhat-y))

#  TODO:  Implémenter la fonction Linear(X, W, b)sur le même modèle que MSE

class Linear(Function):
    @staticmethod
    def forward(ctx,X, W, b=None):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(X, W, b)
        output = X.mm(W.t())
        if b is not None:
            output += b.unsqueeze(0).expand_as(output)
        return output
        
        #  TODO:  Renvoyer la valeur de la fonction


    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
       X, W, b = ctx.saved_tensors
       
       grad_X= grad_W = grad_b= None
       if (W!=None):
       	grad_X = grad_output.mm(W)
       if (X!=None):
      	 grad_W = grad_output.t().mm(X)
       if (b!=None):
       	grad_b = grad_output.sum(0)
       return grad_X, grad_W, grad_b

## Utile dans ce TP que pour le script tp1_gradcheck
mse = MSE.apply
linear = Linear.apply

