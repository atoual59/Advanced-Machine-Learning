
from textloader import *
from generate import *
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from tqdm.autonotebook import tqdm
#  TODO: 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    #  TODO:  Implémenter maskedCrossEntropy sans aucune boucle, la CrossEntropy qui ne prend pas en compte les caractères de padding.
    res=torch.tensor(0.).to(device)
    loss = nn.CrossEntropyLoss(reduce=False)
    tloss =loss(output,target)
    for i in range (len(padcar)):
      ind=padcar[i]
      val=0
      if ind ==0 : ind=1
      if ind!=-1 : 
        val= sum(tloss[i][:ind])/ind
      else : val = sum(tloss[i])/len(tloss.shape[1])
      res += val
    return res/tloss.shape[0]


class RNN(nn.Module):
    #  TODO:  Recopier l'implémentation du RNN (TP 4)
    def __init__(self,batch,dim,hidden_size,output_size):
        super(RNN, self).__init__()
        self.hidden_in=nn.Linear(dim+hidden_size,hidden_size).to(device)
        self.hidden_out=nn.Linear(hidden_size,output_size).to(device)
        self.batch=batch
        self.dim=dim
        self.hidden_size=hidden_size
    def one_step(self,x,h) :
      a=F.tanh(self.hidden_in(torch.cat([x.to(device), h.to(device)],dim=1)))
      a=a.to(device)
      return a
    def forward(self,x):
      x=x.permute(1,0,2)
      h=torch.zeros(x.shape[1],self.hidden_size).to(device)
      res=torch.empty((len(x),x.shape[1],self.hidden_size),dtype=torch.float32).to(device)
      res[0]=h
      for i in range (len(x)) :
        h=self.one_step(x[i],h)
        res[i]=h
      return res
    def decode (self,h):
        return self.hidden_out(h.to(device))


class LSTM(nn.Module):
    #  TODO:  Implémenter un LSTM
    def __init__(self,batch,dim,hidden_size,output_size):
        super(LSTM, self).__init__()
        self.ft=nn.Linear(hidden_size+dim,hidden_size,bias=True)
        self.it=nn.Linear(hidden_size+dim,hidden_size,bias=True)
        self.ot=nn.Linear(hidden_size+dim,hidden_size,bias=True)
        self.sig=nn.Sigmoid()
        self.hidden_in=nn.Linear(dim+hidden_size,hidden_size,bias=True)
        self.hidden_out=nn.Linear(hidden_size,output_size)
        self.batch=batch
        self.dim=dim
        self.hidden_size=hidden_size
    def one_step(self,x,h,c) :
      ft=self.sig(self.ft(torch.cat([h.to(device),x.to(device)],dim=1)))
      it=self.sig(self.it(torch.cat([h.to(device),x.to(device)],dim=1)))
      etape3=F.tanh(self.hidden_in(torch.cat([h.to(device),x.to(device)],dim=1)))
      ct=(ft.to(device) * c.to(device))+(it.to(device) * etape3.to(device))
      return self.sig(self.ot(torch.cat([h.to(device),x.to(device)],dim=1))) * F.tanh(ct),ct
    def forward(self,x):
      x=x.permute(1,0,2)
      h=torch.zeros(x.shape[1],self.hidden_size).to(device)
      c=torch.zeros(x.shape[1],self.hidden_size).to(device)
      res=torch.empty((len(x),x.shape[1],self.hidden_size),dtype=torch.float32).to(device)
      res[0]=h
      for i in range (len(x)) :
        h,c=self.one_step(x[i],h,c)
        res[i]=h
      return res
    def decode (self,h):
        return self.hidden_out(h)



class GRU(nn.Module):
    #  TODO:  Implémenter un GRU
    def __init__(self,batch,dim,hidden_size,output_size):
        super(GRU, self).__init__()
        self.zt=nn.Linear(hidden_size+dim,hidden_size,bias=False)
        self.rt=nn.Linear(hidden_size+dim,hidden_size,bias=False)
        self.sig=nn.Sigmoid()
        self.hidden_in=nn.Linear(dim+hidden_size,hidden_size,bias=False)
        self.hidden_out=nn.Linear(hidden_size,output_size)
        self.batch=batch
        self.dim=dim
        self.hidden_size=hidden_size
    def one_step(self,x,h) :
        etape1=self.sig(self.zt(torch.cat([ h.to(device),x.to(device)],dim=1)))
        etape2=self.sig(self.rt(torch.cat([h.to(device),x.to(device)],dim=1)))
        return (1-etape1.to(device)) * h.to(device)+etape1.to(device) * F.tanh(self.hidden_in(torch.cat([etape2.to(device)*h.to(device), x.to(device)],dim=1)))
    def forward(self,x):
      x=x.permute(1,0,2)
      h=torch.zeros(x.shape[1],self.hidden_size).to(device)
      res=torch.empty((len(x),x.shape[1],self.hidden_size),dtype=torch.float32).to(device)
      res[0]=h
      for i in range (len(x)) :
        h=self.one_step(x[i],h)
        res[i]=h
      return res
    def decode (self,h):
        return self.hidden_out(h)

class State :
    def __init__(self ,model,optim):
        self.model = model
        self.optim = optim
        self.epoch , self.iteration = 0,0


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




#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot
""""
### test MaskedEntropy
input = torch.randn(6, 5, requires_grad=True)
print(input)
target = torch.empty(6, dtype=torch.long).random_(5)
print(target)
print(maskedCrossEntropy(input,target,3))
"""

#### TEST generation avec embeddings


f = open('trump_full_speech.txt','r')
text=f.read()
BATCH_SIZE=100
CLASSES=98
DIM_INPUT=100


savepath = Path("model.pch")
lossVal=[]
ds = TextDataset(text)
data_train = DataLoader(ds,collate_fn=pad_collate_fn, batch_size=BATCH_SIZE)
embeded=nn.Embedding(182, DIM_INPUT)
embeded.to(device)
writer = SummaryWriter()
hidden_size = 100
learning_rate = 10e-3
num_epochs = 50
if savepath.is_file ():
    print("true")
    with savepath.open("rb") as fp :
        state = torch.load(fp)
else :
    model =LSTM(BATCH_SIZE,DIM_INPUT,hidden_size,CLASSES)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)
    #lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    state = State(model,optimizer)
for epoch in range(state.epoch,num_epochs) :
  tot=0
  tloss=0
  t = tqdm(iter(data_train), total=len(data_train), dynamic_ncols=True)
  for inputs in t :
    if(inputs.shape[1]==BATCH_SIZE):
      tot+=1
      labels= inputs[1:].to(device)
      inputs = inputs[:-1]
      inputs=inputs.to(device)
      labels=labels.to(device)
      emb= embeded(inputs)
      state.optim.zero_grad()
      outputs=state.model.forward(emb)
      decode=state.model.decode(outputs)
      decode=decode.permute(1,2,0)
      indxList=[]
      inputs=inputs.permute(1,0)
      for x in inputs :
        indx=-1
        try :
          indx =x.tolist().index(0)
          indxList.append(indx)
        except ValueError:
          continue
      loss=maskedCrossEntropy(decode.to(device),labels.to(device), indxList).to(device)
      tloss+=loss.item()
      loss.backward()
      state.optim.step()
      state.iteration += 1
  with savepath.open("wb") as fp: 
        state.epoch = epoch + 1 
        torch.save(state ,fp)
  tloss=tloss/tot
  lossVal.append(tloss)
  print("train loss for epoch: ",epoch ," is : ", tloss)
  #lr_sched.step()
  #writer.add_scalar('Loss/train',tloss, epoch)



result=generate(state.model, embeded, state.model.decode, id2lettre[EOS_IX], start="When was the last ", maxlen=200)
print(code2string(result))
plt.plot(np.arange(50),lossVal)