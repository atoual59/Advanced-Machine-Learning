from textloader import  string2code, id2lettre,code2string
import math
import torch

#  TODO:  Ce fichier contient les différentes fonction de génération

def generate(rnn, emb, decoder, eos, start="", maxlen=200): ##collate_fn-> elle renvoie la matrice embided
    """  Fonction de génération (l'embedding et le decodeur être des fonctions du rnn). Initialise le réseau avec start (ou à 0 si start est vide) et génère une séquence de longueur maximale 200 ou qui s'arrête quand eos est généré.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    res=[]
    debut=torch.tensor([0]).to(device)
    if (start!=""):
        debut=string2code(start).to(device)
    while (code2string(debut)[-1]!=eos and maxlen>0):
      em=emb(debut).to(device)
      em=em.reshape(1,em.shape[0],em.shape[1])
      out=rnn.forward(em).to(device)
      output=decoder(out[-1])
      _, predicted = torch.max(output.data, 1)
      res.append(predicted.item())
      debut=torch.cat((debut[1:],predicted),0)
      maxlen-=1
    return res


    #  TODO:  Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits (logarithme de probabilité à une constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles


def generate_beam(rnn, emb, decoder, eos, k, start="hey", maxlen=200):
    """
        Génere une séquence en beam-search : à chaque itération, on explore pour chaque candidat les k symboles les plus probables; puis seuls les k meilleurs candidats de l'ensemble des séquences générées sont conservés (au sens de la vraisemblance) pour l'itération suivante.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * k : le paramètre du beam-search
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    #  TODO:  Implémentez le beam Search


# p_nucleus
def p_nucleus(decoder, alpha: float):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        * decoder: renvoie les logits étant donné l'état du RNN
        * alpha (float): masse de probabilité à couvrir
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
           * h (torch.Tensor): L'état à décoder
        """
        #  TODO:  Implémentez le Nucleus sampling ici (pour un état s)
    return compute
