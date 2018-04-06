

import model
import h5py
import keras
import random
import argparse


parser = argparse.ArgumentParser(description='NMT Keras')
parser.add_argument('--dataset', type=str,  default="./data/nmt_hi_en_prepped.h5" , help='Path to HDF5 file containg the translation data')
parser.add_argument('--weights_path', type=str,  default="./weights/KerasAttentionNMT_1.h5" , help='Path to Weights checkpoint')

args = parser.parse_args()

hf = h5py.File( args.dataset , 'r')


enc_seq_length = 35 
enc_vocab_size = 40005 
dec_seq_length = 35 
dec_vocab_size = 40005 



inp_x = hf['eng_sent_mat'][:  , : enc_seq_length ]
inp_cond_x = hf['hin_sent_mat'][:  , : dec_seq_length ]
out_y = hf['hin_sent_mat'][:  , 1 : dec_seq_length+1 ]




tr_data = range( inp_x.shape[0] )
random.shuffle(tr_data)

def load_data( batchSize=32 ):
	while True:
		for i in range( 0 , len(tr_data)-batchSize , batchSize ):
			inds = tr_data[ i : i + batchSize  ]
			yield [ inp_x[inds ] , inp_cond_x[ inds ] ] , keras.utils.to_categorical( out_y[ inds ] , num_classes=dec_vocab_size)


tr_gen =  load_data( batchSize=32 )


m = model.getModel()

for ep in range( 100 ):
	print "Epoch" , ep
	m.fit_generator( tr_gen , steps_per_epoch=1000 , epochs=1   )
	m.save_weights( args.weights_path + "." + str(ep) )
	m.save_weights( args.weights_path )
	
print "Training is finished"


