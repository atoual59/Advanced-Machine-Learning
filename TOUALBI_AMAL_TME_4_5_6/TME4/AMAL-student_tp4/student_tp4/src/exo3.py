from utils import RNN, device,  ForecastMetroDataset

from torch.utils.data import  DataLoader
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 30
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32

PREDICTED_SIZE=15
PATH = ""


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch", "rb"))
ds_train = ForecastMetroDataset(
    matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = ForecastMetroDataset(
    matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

#  TODO:  Question 3 : Prédiction de séries temporelles


writer = SummaryWriter()
hidden_size = 20
learning_rate = 1e-3

model =RNN(BATCH_SIZE,DIM_INPUT*CLASSES,hidden_size,DIM_INPUT*CLASSES)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 100
for epoch in range(num_epochs) :
  tloss=0
  t = tqdm(iter(data_train), total=len(data_train), dynamic_ncols=True)
  for inputs,labels in t :
  	if(len(inputs)==BATCH_SIZE):
	    inputs, labels = inputs.to(device), labels.to(device)
	    inputs=inputs.reshape(inputs.shape[0],inputs.shape[1],-1)
	    labels=labels.reshape(inputs.shape[0],inputs.shape[1],-1)
	    labels=labels.permute(1,0,2)
	    optimizer.zero_grad()
	    outputs=model.forward(inputs)
	    loss=torch.tensor(0.,)
	    for i in range(PREDICTED_SIZE,len(outputs)) :	
	    	decode=model.decode(outputs[i])
	    	loss+=criterion(decode,labels[i])
	    	tloss+=loss.item()
	    loss.backward()
	    optimizer.step() 
  tloss=tloss
  print("train loss for epoch: ",epoch ," is : ", tloss)
  writer.add_scalar('Loss/train',tloss, epoch)
  tloss=0
  with torch.no_grad():
    t = tqdm(iter(data_test), total=len(data_test), dynamic_ncols=True)
    for inputs,labels in t :
    	if(len(inputs)==BATCH_SIZE):
	      inputs, labels = inputs.to(device), labels.to(device)
	      inputs=inputs.reshape(inputs.shape[0],inputs.shape[1],-1)
	      labels=labels.reshape(inputs.shape[0],inputs.shape[1],-1)
	      labels=labels.permute(1,0,2)
	      optimizer.zero_grad()
	      outputs=model.forward(inputs)
	      loss=torch.tensor(0.)
	      for i in range(PREDICTED_SIZE,len(outputs)) :	
	      	decode=model.decode(outputs[i])
	      	loss+=criterion(decode,labels[i])
	      	tloss+=loss.item()
  tloss=tloss
  print("test loss for epoch: ",epoch ," is : ", tloss)
  writer.add_scalar('Loss/test',tloss, epoch)
  #print('Accuracy in test', (100.0 * ok / tot))




