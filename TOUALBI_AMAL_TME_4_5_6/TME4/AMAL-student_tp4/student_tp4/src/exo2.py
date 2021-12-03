from utils import RNN, device,SampleMetroDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence
writer = SummaryWriter()

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32

PATH = ""


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test=SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)
data_train = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE,shuffle=False)



hidden_size = 15
learning_rate = 1e-3

model =RNN(BATCH_SIZE,DIM_INPUT,hidden_size,CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 100
for epoch in range(num_epochs) :
  tloss=0
  t = tqdm(iter(data_train), total=len(data_train), dynamic_ncols=True)
  for inputs,labels in t :
  	if(len(inputs)==BATCH_SIZE):
	    inputs, labels = inputs.to(device), labels.to(device)
	    optimizer.zero_grad()
	    outputs=model.forward(inputs)
	    outputs=outputs[-1]
	    decode=model.decode(outputs)
	    loss=criterion(decode,labels)
	    tloss+=loss.item()
	    loss.backward()
	    optimizer.step() 
  tloss=tloss/len(data_train)
  print("train loss for epoch: ",epoch ," is : ", tloss)
  writer.add_scalar('Loss/train',tloss, epoch)
  tloss=0
  ok=0
  tot=0
  with torch.no_grad():
    t = tqdm(iter(data_test), total=len(data_test), dynamic_ncols=True)
    for inputs,labels in t :
    	if(len(inputs)==BATCH_SIZE):
	      inputs, labels = inputs.to(device), labels.to(device)
	      optimizer.zero_grad()
	      outputs=model.forward(inputs)
	      outputs=outputs[-1]
	      decode=model.decode(outputs)
	      loss=criterion(decode,labels)
	      _, predicted = torch.max(decode.data, 1)
	      tot += labels.size(0)
	      ok += (predicted == labels).sum().item()
	      tloss+=loss.item()
  tloss=tloss/len(data_test)
  print("test loss for epoch: ",epoch ," is : ", tloss)
  writer.add_scalar('Loss/test',tloss, epoch)
  #print('Accuracy in test', (100.0 * ok / tot))







