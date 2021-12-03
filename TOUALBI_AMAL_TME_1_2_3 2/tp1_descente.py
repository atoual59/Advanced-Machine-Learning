import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context


# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 3)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3)
b = torch.randn(3)


ep=[0.0005]
writer = SummaryWriter()
for epsilon in ep : 
    print("epsilon = ",epsilon)
    loss=0
    for n_iter in range(100):
        ##  TODO:  Calcul du forward (loss)
        ctx1=Context()
        output=Linear.forward(ctx1,x,w.t(),b)
        ctx2=Context()
        loss=torch.mean(MSE.forward(ctx2,output,y))
        # `loss` doit correspondre au coût MSE calculé à cette itération
        # on peut visualiser avec
        # tensorboard --logdir runs/
        titre='Loss/train pour epsilon = '+str(epsilon)
        writer.add_scalar(titre, loss, n_iter)

        # Sortie directe
        print(f"Itérations {n_iter}: loss {loss}")
        ##  TODO:  Calcul du backward (grad_w, grad_b)
        grad_output=MSE.backward(ctx2,loss)
        _,grad_w,grad_b = Linear.backward(ctx1,grad_output[0])
        ##  TODO:  Mise à jour des paramètres du modèle
        w-=epsilon*grad_w.t()
        b-=epsilon*grad_b.t()

