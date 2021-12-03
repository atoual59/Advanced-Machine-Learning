from pathlib import Path
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
from tqdm.autonotebook import tqdm


# Téléchargement des données

from datamaestro import prepare_dataset
ds = prepare_dataset("com.lecun.mnist");
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Pour visualiser
# Les images doivent etre en format Channel (3) x Hauteur x Largeur
images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()/255.
# Permet de fabriquer une grille d'images
images = make_grid(images)
# Affichage avec tensorboard
writer.add_image(f'samples', images, 0)


savepath = Path("model.pch")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size=10
num_epoch=10
#  TODO: 

######   CREATION DATASET  ##########

class MonDataset(Dataset):
	def __init__( self ,images,labels): 
		self.images=images.astype('float32')
		self.labels=labels.astype('long')
	def __getitem__( self ,index ):
		return (self.images[index]/255),self.labels[index]
	def __len__(self):
		return self.images.shape[0]

trainloader = DataLoader(MonDataset(train_images,train_labels) , shuffle=True, batch_size=batch_size)
testloader = DataLoader(MonDataset(test_images,test_labels) , shuffle=True, batch_size=batch_size)



"""
#####  afin de tester le loader   ##########
print(next(iter(trainloader))[0][0])
print(next(iter(trainloader))[0].shape)
t = tqdm(iter(trainloader), total=len(trainloader), dynamic_ncols=True)
for d,l in t:
	print(d,l)



######   AUTO ENCODEUR  ##########


class AutoEncoder(torch.nn.Module):
	 def __init__(self):
	 	super(AutoEncoder, self).__init__()
	 	self.fc1=torch.nn.Linear(784,500)
	 	self.fc2=torch.nn.Linear(500,784)
	 	self.sig=torch.nn.Sigmoid()
	 def forward(self, x):	
	 	x=self.fc1(x)
	 	x=F.relu(x)
	 	x=self.fc2(x)
	 	x=self.sig(x)
	 	return x


net = AutoEncoder()
learning_rate = 1e-2
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
mse = torch.nn.MSELoss()
print(net)
titre='train/loss avec eps= '+ str(learning_rate)
for epoch in range(num_epoch):
	tloss=0
	t = tqdm(iter(trainloader), total=len(trainloader), dynamic_ncols=True)
	for d,l in t:
		optimizer.zero_grad()
		d = d.view(d.size()[0], -1)
		outputs = net(d)
		loss = mse(outputs, d) 
		loss.backward()
		optimizer.step()
		tloss+=loss.item()
		#i+=1
	print(tloss)
	writer.add_scalar(titre, tloss, epoch)

"""
#### Classifieur avec GPU et CHECKPOINTING  #######
class State :
	def __init__(self ,model,optim):
		self.model = model
		self.optim = optim
		self.epoch , self.iteration = 0,0
class Net(torch.nn.Module):
	 def __init__(self):
	 	super(Net, self).__init__()
	 	self.fc1=torch.nn.Linear(28*28,120)
	 	self.fc2 = nn.Linear(120, 84)
	 	self.fc3 = nn.Linear(84, 10)
	 def forward(self, x):
	 	x = x.view(x.size()[0], -1)
	 	x = F.relu(self.fc1(x))
	 	x = F.relu(self.fc2(x))
	 	x = self.fc3(x)
	 	return x

criterion = nn.CrossEntropyLoss()
writer = SummaryWriter()
if savepath.is_file ():
	print("true")
	with savepath.open("rb") as fp :
		state = torch.load(fp)
else :
	model = Net()
	model = model.to(device) 
	learning_rate = 1e-2
	optim = torch.optim.SGD(model.parameters(), lr=learning_rate)
	state = State(model,optim)
for epoch in range(state.epoch,num_epoch):
	tloss=0
	t = tqdm(iter(trainloader), total=len(trainloader), dynamic_ncols=True)
	for data in t:
		inputs, labels = data
		inputs, labels = inputs.to(device), labels.to(device)
		state.optim.zero_grad() 
		outputs = state.model(inputs)
		loss = criterion(outputs, labels)
		tloss+=loss.item()
		loss.backward()
		state.optim.step() 
		state.iteration += 1
	with savepath.open("wb") as fp: 
		state.epoch = epoch + 1 
		torch.save(state ,fp)
	tloss=tloss/len(trainloader)
	print("train loss for epoch: ",epoch ," is : ", tloss)
	writer.add_scalar('Loss/train',tloss, epoch)
	tloss=0.0
	ok = 0
	tot = 0
	with torch.no_grad():
		t = tqdm(iter(testloader), total=len(testloader))
		for data in t:
			inputs, labels = data
			inputs, labels = inputs.to(device), labels.to(device)
			outputs = state.model(inputs)
			loss = criterion(outputs, labels)
			_, predicted = torch.max(outputs.data, 1)
			tot += labels.size(0)
			ok += (predicted == labels).sum().item()
			writer.close()
			tloss += loss.item()
	tloss=tloss/len(testloader)
	print("test loss for epoch: ",epoch ," is : ", tloss)
	print('Accuracy in test', (100.0 * ok / tot))
	writer.add_pr_curve('pr_curve', labels, predicted, epoch)
	writer.add_scalar('Loss/test', tloss, epoch)
	writer.add_scalar('Accuracy/test', (100.0 * ok / tot), epoch)






