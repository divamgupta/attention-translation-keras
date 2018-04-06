

from tqdm import tqdm
import numpy as np


import numpy as np
from functools import partial
from threading import Thread

import json
import numpy as np


# from nltk.tokenize import word_tokenize

# def tokenise( line ):
# 	return word_tokenize(str(line).lower())
nlp = None

def get_nlp():
    global nlp 
    if nlp is None:
        import spacy                           # See "Installing spaCy"
        nlp = spacy.load('en')                 # You are here.


def spacy_tokenize(line):
    get_nlp()
    doc = nlp( line )  
    p =  [ str(d).lower() for d in doc]
    return p

def tokenise( line , tokenizer_fn=spacy_tokenize ,  startEndTokens=False  ):
    p = tokenizer_fn(line)
    if startEndTokens:
        p = ['<start>'] + p + ['<end>']
    return p


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.wordFreqs = {}
        self.idx = 1 # id 0 is reserved for padding
        self.add_word( '<unk>' )
        self.add_word( '<start>' )
        self.add_word( '<end>' )

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            self.wordFreqs[ word ]  = 0
        self.wordFreqs[ word ] += 1

    def add_words(self , words ):
        for w in words:
            self.add_word( w )

    def add_sentence(self , sentence ):
        words = tokenise( sentence )
        for w in words:
            self.add_word( w )

    def keepTopK( self , k ):
        wordFreqs = self.wordFreqs.items()
        wordFreqs = sorted( wordFreqs , key=lambda x:x[1] , reverse=True)
        wordFreqs = wordFreqs[:k]
        words = [ w[0] for w in  wordFreqs ]

        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

        for word in ['<unk>' , '<start>'  , '<end>' ] + words:
            if not word in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def getDicts( self ):
        J = { "idx":self.idx  , "idx2word":self.idx2word , "word2idx":self.word2idx , "wordFreqs":self.wordFreqs    } 
        return J


    def save( self, fname_prefix ):
        open( fname_prefix + "word2idx.json" , "wb" ).write( json.dumps( self.word2idx )  )
        open( fname_prefix + "idx2word.json" , "wb" ).write( json.dumps( self.idx2word )  )
        J = { "idx":self.idx  , "idx2word":self.idx2word , "word2idx":self.word2idx , "wordFreqs":self.wordFreqs    } 
        open( fname_prefix + "everything.json" , "wb" ).write( json.dumps( J )  )

    def load(self):
        d = open( fname_prefix + "everything.json"  ).read()
        self.idx  = d['ids']
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.wordFreqs  = wordFreqs


def getSentencesMat( sentences , vocab , maxSentenceL=None , padding='right' , startEndTokens=False , tokenizer_fn=spacy_tokenize ):

    tokenised = [ tokenise(s , startEndTokens=startEndTokens ,tokenizer_fn=tokenizer_fn ) for s in sentences ]

    if maxSentenceL is None:
        maxSentenceL = max([ len(s) for s in tokenised  ])
    sentencesMat = np.zeros((len(sentences) , maxSentenceL)).astype('int64')
    for i , sen in enumerate(tokenised) :
        ids = [ vocab(w) for w in sen  ]
        if padding == 'right':
            ids = ids[:maxSentenceL]
        else:
            ids = ids[-1*maxSentenceL:]

        if len(ids) < maxSentenceL:
            if padding == 'right':
                ids = ids + [0]*(maxSentenceL - len(ids) )
            else:
                ids =  [0]*(maxSentenceL - len(ids) ) + ids
        sentencesMat[i] = np.array(ids)
    return sentencesMat


       