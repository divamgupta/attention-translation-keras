# -*- coding: utf-8 -*-

import model
import h5py
import keras
import random
import argparse
import json

import numpy as np 


parser = argparse.ArgumentParser(description='NMT Keras')
parser.add_argument('--dataset', type=str,  default="./data/nmt_hi_en_prepped.h5" , help='Path to HDF5 file containg the translation data')
parser.add_argument('--weights_path', type=str,  default="./weights/KerasAttentionNMT_1.h5" ,  help='Path to Weights checkpoint')

args = parser.parse_args()

hf = h5py.File( args.dataset , 'r')
m  = model.getModel()
m.load_weights( args.weights_path )

en_vocab = json.loads(hf['en_vocab'].value)
hi_vocab = json.loads(hf['hi_vocab'].value)


def predict( sent ):
	words = sent.split(' ')
	words = ['<start>'] + words + ['<end>']
	words_id = []

	for w in words:
		if w in en_vocab['word2idx']:
			words_id.append( en_vocab['word2idx'][w] )
		else:
			words_id.append( en_vocab['word2idx']['<unk>'] )
	words = words_id	
	

	ret = ""

	m_input = [ np.zeros((1,35)) , np.zeros((1,35)) ]
	for i , w in enumerate( words ):
		m_input[0][0 , i ] = w
	m_input[1][0,0] = hi_vocab['word2idx']['<start>']

	for w_i in range(1,35):
		out = m.predict( m_input )
		out_w_i = out[0][w_i-1].argmax()
		
		if out_w_i == 0:
			continue
		
		ret +=  hi_vocab['idx2word'][str(out_w_i)] + " "
		m_input[1][0,w_i] = out_w_i

	return ret



while True:
	print "Enter a sentence : "
	sent = raw_input()
	print predict( sent ).encode('utf-8')

	print "==============="


