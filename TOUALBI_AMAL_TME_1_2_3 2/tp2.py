import torch
from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
from tqdm import tqdm
import torch.nn.functional as F
import datetime

writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

data = datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()
datax = torch.tensor(datax,dtype=torch.float,requires_grad=True)
datay = torch.tensor(datay,dtype=torch.float,requires_grad=True).reshape(-1,1)
w=torch.randn(datax.shape[1], datay.shape[1],requires_grad=True)
b=torch.randn(1,1,requires_grad=True)
writer = SummaryWriter()

##descente batch
"""
num_epoch=20
ep=1e-8
for epoch in range(num_epoch):
	tloss=0
	yhat=datax@w +b
	tloss=torch.mean((yhat-datay)**2)
	titre='Loss/train'+str(ep)
	writer.add_scalar(titre, tloss.item(), epoch)
	print(f"Itérations {epoch}: loss {tloss}")
	tloss.backward()
	w.data=w-ep*w.grad
	w.retain_grad()
	b.data=b-ep*b.grad
	b.retain_grad()

##descente stockastique	

num_epoch=15
ep=1e-13
titre='Loss/train'+str(ep)
w=torch.randn(datax.shape[1], datay.shape[1],requires_grad=True)
b=torch.randn(1,1,requires_grad=True)
for epoch in range(num_epoch):
	loss=0
	for x in datax:
		yhat=x@w +b
		tloss=torch.mean((yhat-datay)**2)
		loss+=tloss
		tloss.backward()
		w.data=w-ep*w.grad
		w.retain_grad()
		b.data=b-ep*b.grad
		b.retain_grad()
	writer.add_scalar(titre, loss, epoch)
	print(f"Itérations {epoch}:\n loss {loss}")


## descente mini-batch

num_epoch=10
ep=1e-13
w=torch.randn(datax.shape[1], datay.shape[1],requires_grad=True)
b=torch.randn(1,1,requires_grad=True)
batch_size=2
titre='Loss/train'+str(ep)
for epoch in range(num_epoch):
	loss=0
	tmp=0
	for x in range (batch_size,datax.shape[0],batch_size):
		yhat=datax[tmp:x]@w +b
		tloss=torch.mean((yhat-datay[tmp:x])**2)
		tmp=x
		loss+=tloss
		tloss.backward()
		w.data=w-ep*w.grad
		w.retain_grad()
		b.data=b-ep*b.grad
		b.retain_grad()
	writer.add_scalar(titre, loss, epoch)
	print(f"Itérations {epoch}:\n loss {loss}")

"""
# TODO: 

############    Reseau à  deux couches    ####################


num_epoch=30
class Net(torch.nn.Module):
	 def __init__(self):
	 	super(Net, self).__init__()
	 	self.fc1=torch.nn.Linear(13,13)
	 	self.fc2=torch.nn.Linear(13,1)
	 def forward(self, x): 	 	
	 	x=self.fc1(x)
	 	x=F.tanh(x)
	 	x=self.fc2(x)
	 	return x


net =Net()
learning_rate = 1e-5
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
mse = torch.nn.MSELoss()
titre='Loss/train'+str(learning_rate)
for epoch in range(num_epoch):
	tloss=0
	i=0
	for inputs in datax :
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = mse(outputs, datay[i]) 
		loss.backward()
		optimizer.step()
		tloss+=loss.item()
		i+=1
	print(tloss)
	writer.add_scalar(titre, tloss, epoch)


