import string
import unicodedata
import torch
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader
from tqdm.autonotebook import tqdm
from pathlib import Path
from utils import RNN, device
import torch.nn.functional as F
import numpy as np

## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)
    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        return t[:-1],t[1:]



#  TODO: 
class State :
    def __init__(self ,model,optim):
        self.model = model
        self.optim = optim
        self.epoch , self.iteration = 0,0



class RNNTEXT(nn.Module):
    #  TODO:  Implémenter comme décrit dans la question 1
    def __init__(self,batch,dim,hidden_size,output_size,dim_encoded):
        super(RNNTEXT, self).__init__()
        self.fc=nn.Linear(dim,dim_encoded)
        self.rnn=RNN(batch,dim_encoded,hidden_size,output_size)
    def forward(self,x):
        x=self.fc(x)
        x=self.rnn(x)
        return x
    def decode (self,h):
        return self.rnn.decode(h)

f = open('trump_full_speech.txt','r')
text=f.read()

BATCH_SIZE=32
SEQ_LENGTH=20
CLASSES=96
DIM_INPUT=70

savepath = Path("model.pch")

ds = TrumpDataset(text,maxlen=SEQ_LENGTH)
data_train = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

writer = SummaryWriter()
hidden_size = 20
learning_rate = 1e-3
num_epochs = 50
criterion = nn.CrossEntropyLoss()
if savepath.is_file ():
    print("true")
    with savepath.open("rb") as fp :
        state = torch.load(fp)
else :
    model =RNNTEXT(BATCH_SIZE,CLASSES,hidden_size,CLASSES,DIM_INPUT)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    state = State(model,optimizer)
for epoch in range(state.epoch,num_epochs) :
  tloss=0
  t = tqdm(iter(data_train), total=len(data_train), dynamic_ncols=True)
  for inputs,labels in t :
    if(len(inputs)==BATCH_SIZE):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs=F.one_hot(inputs, num_classes=CLASSES).type(torch.float)
        labels=labels.permute(1,0)
        state.optim.zero_grad()
        outputs=state.model.forward(inputs)
        loss=torch.tensor(0.,)
        for i in range(len(outputs)) :   
            decode=state.model.decode(outputs[i])
            loss+=criterion(decode,labels[i])
            tloss+=loss.item()
        loss.backward()
        state.optim.step() 
        state.iteration += 1
  with savepath.open("wb") as fp: 
        state.epoch = epoch + 1 
        torch.save(state ,fp)
  tloss=tloss/len(data_train)
  print("train loss for epoch: ",epoch ," is : ", tloss)
  writer.add_scalar('Loss/train',tloss, epoch)


res=np.zeros((BATCH_SIZE,SEQ_LENGTH))
with torch.no_grad():
    inputs,_=next(iter(data_train))
    ##inputs=torch.zeros(BATCH_SIZE,SEQ_LENGTH,dtype=torch.int64)
    inputs[-1][-1]=93.
    inputs=F.one_hot(inputs, num_classes=CLASSES).type(torch.float) 
    for i in range (SEQ_LENGTH):
        state.optim.zero_grad()
        outputs=state.model.forward(inputs)
        decode=state.model.decode(outputs[-1])
        _, predicted = torch.max(decode.data, 1)
        for j in range(BATCH_SIZE):
            res[j,i]=predicted.tolist()[j]
        temp=torch.empty(SEQ_LENGTH,BATCH_SIZE,CLASSES)
        inputs=inputs.permute(1,0,2)
        for n in range(SEQ_LENGTH):
            if(n==SEQ_LENGTH-1):
                decode=state.model.decode(outputs[n])
                predicted = torch.argmax(decode.data,dim=1)
                a=F.one_hot(predicted, num_classes=CLASSES) 
                temp[n]=a
            else:
                temp[n]=inputs[n]
        inputs=temp
        inputs=inputs.permute(1,0,2)

    for m in range (BATCH_SIZE):
        print(code2string(res[m]))
        print("\n")
        


        


